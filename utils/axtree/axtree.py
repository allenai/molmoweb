"""
Accessibility tree extraction and flattening for Playwright pages.

Uses Chrome DevTools Protocol to:
1. Mark DOM elements with unique bid attributes (via injected JS)
2. Extract the accessibility tree per frame and merge them
3. Extract element properties (visibility, bbox, clickable) from DOM snapshot
4. Flatten the tree to a text string for LLM consumption
5. Clean up temporary attributes

No external dependencies beyond playwright and numpy/PIL.
"""
import base64
import io
import logging
import os
import re

import numpy as np
import PIL.Image
import playwright.sync_api

logger = logging.getLogger(__name__)

BID_ATTR = "bid"
VIS_ATTR = "molmoweb_visibility_ratio"
SOM_ATTR = "molmoweb_set_of_marks"
EXTRACT_OBS_MAX_TRIES = 5

_JS_DIR = os.path.join(os.path.dirname(__file__), "javascript")
_BID_REGEX = re.compile(r"^molmoweb_id_([a-zA-Z0-9]+)\s?(.*)")

IGNORED_ROLES = {"LineBreak"}
IGNORED_PROPERTIES = frozenset({
    "editable", "readonly", "level", "settable", "multiline", "invalid", "focusable",
})

_SKIP_URL_PREFIXES = (
    "chrome-error://", "chrome://", "chrome-extension://", "devtools://",
    "edge://", "data:", "blob:", "file://",
)
_AD_MARKERS = (
    "ads.", "ad.", "advertising.", "adserver.", "analytics.", "tracking.", "track.",
    "pixel.", "beacon.", "doubleclick", "googlesyndication", "amazon-adsystem",
    "pubmatic", "rubicon", "criteo", "adsystem",
)


class MarkingError(Exception):
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_axtree(page: playwright.sync_api.Page, lenient: bool = True) -> tuple[dict, dict]:
    """Extract merged axtree and extra element properties from a page.

    Returns:
        (axtree, extra_properties) where axtree is the merged CDP accessibility
        tree dict and extra_properties maps bid -> {visibility, bbox, clickable, set_of_marks}.
    """
    _mark_elements(page, lenient=lenient)
    try:
        dom = _extract_dom_snapshot(page)
        axtree = _extract_merged_axtree(page)
        extra = _extract_extra_properties(dom)
    finally:
        _unmark_elements(page)
    return axtree, extra


def extract_screenshot(page: playwright.sync_api.Page, scale_factor: float = 1.0) -> np.ndarray:
    """Capture a screenshot via CDP. Returns RGB numpy array."""
    cdp = page.context.new_cdp_session(page)
    vp = page.viewport_size
    if vp is None:
        dims = page.evaluate("() => ({width: window.innerWidth, height: window.innerHeight})")
    else:
        dims = {"width": vp["width"], "height": vp["height"]}

    original = {"width": dims["width"], "height": dims["height"], "deviceScaleFactor": 1.0, "mobile": False}
    cdp.send("Emulation.setDeviceMetricsOverride", {**original, "deviceScaleFactor": scale_factor})
    data = cdp.send("Page.captureScreenshot", {"format": "png"})["data"]
    cdp.send("Emulation.setDeviceMetricsOverride", original)
    cdp.detach()

    img = PIL.Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    return np.array(img)


def flatten_axtree_to_str(
    axtree: dict,
    extra_properties: dict = None,
    filter_visible_only: bool = True,
    filter_with_bid_only: bool = True,
    with_clickable: bool = True,
    skip_generic: bool = True,
    with_visible: bool = False,
    with_center_coords: bool = False,
    with_bounding_box_coords: bool = False,
    coord_decimals: int = 0,
    remove_redundant_static_text: bool = True,
) -> str:
    """Convert an axtree dict to an indented text representation for LLMs."""
    if extra_properties is None:
        extra_properties = {}

    if "nodes" not in axtree or not axtree["nodes"]:
        return ""

    node_id_to_idx = {node["nodeId"]: idx for idx, node in enumerate(axtree["nodes"])}

    def _bid_attrs(bid):
        skip = False
        attrs = []
        if bid is None:
            if filter_with_bid_only:
                skip = True
        elif bid in extra_properties:
            p = extra_properties[bid]
            if filter_visible_only and p["visibility"] < 0.5:
                skip = True
            if with_clickable and p["clickable"]:
                attrs.append("clickable")
            if with_visible and p["visibility"] >= 0.5:
                attrs.append("visible")
            if with_center_coords and p["bbox"]:
                x, y, w, h = p["bbox"]
                c = f".{coord_decimals}f"
                attrs.append(f'center="({(x+w/2):{c}},{(y+h/2):{c}})"')
            if with_bounding_box_coords and p["bbox"]:
                x, y, w, h = p["bbox"]
                c = f".{coord_decimals}f"
                attrs.append(f'box="({x:{c}},{y:{c}},{x+w:{c}},{y+h:{c}})"')
        return skip, attrs

    def _dfs(idx: int, depth: int, parent_filtered: bool, parent_name: str) -> str:
        node = axtree["nodes"][idx]
        role = node["role"]["value"]
        skip = False
        filtered = False
        name = ""
        result = ""

        if role in IGNORED_ROLES or "name" not in node:
            skip = True
        else:
            name = node["name"]["value"]
            value = node["value"]["value"] if "value" in node and "value" in node["value"] else None
            bid = node.get("molmoweb_id")

            props = []
            for p in node.get("properties", []):
                if "value" not in p or "value" not in p["value"]:
                    continue
                pn, pv = p["name"], p["value"]["value"]
                if pn in IGNORED_PROPERTIES:
                    continue
                elif pn in ("required", "focused", "atomic"):
                    if pv:
                        props.append(pn)
                else:
                    props.append(f"{pn}={repr(pv)}")

            if skip_generic and role in ("generic", "group") and not props:
                skip = True

            if role == "StaticText":
                if parent_filtered or (remove_redundant_static_text and name in parent_name):
                    skip = True
            else:
                filtered, extra_attrs = _bid_attrs(bid)
                skip = skip or filtered
                props = extra_attrs + props

            if not skip:
                s = f"{role} {repr(name.strip())}" if name or role != "generic" else role
                if bid is not None:
                    s = f"[{bid}] " + s
                if value is not None:
                    s += f" value={repr(value)}"
                if props:
                    s += ", ".join([""] + props)
                result = "\t" * depth + s
            else:
                result = ""

        parts = [result] if result else []
        for child_id in node["childIds"]:
            if child_id not in node_id_to_idx or child_id == node["nodeId"]:
                continue
            child = _dfs(
                node_id_to_idx[child_id],
                depth if skip else depth + 1,
                filtered, name,
            )
            if child:
                parts.append(child)

        return "\n".join(parts)

    return _dfs(0, 0, False, "")


# ---------------------------------------------------------------------------
# Internal: DOM marking / unmarking
# ---------------------------------------------------------------------------

def _is_skip_frame(url: str) -> bool:
    if not url or url in ("about:blank", "about:srcdoc"):
        return True
    low = url.lower()
    if low.startswith(_SKIP_URL_PREFIXES) or "chromewebdata" in low:
        return True
    if "captcha" in low or "recaptcha" in low:
        return True
    return any(m in low for m in _AD_MARKERS)


def _load_js(name: str) -> str:
    with open(os.path.join(_JS_DIR, name)) as f:
        return f.read()


def _mark_elements(page, tags_to_mark="standard_html", lenient=False):
    js = _load_js("frame_mark_elements.js")

    def _recurse(frame, frame_bid: str):
        if _is_skip_frame(frame.url):
            return
        msgs = frame.evaluate(js, [frame_bid, BID_ATTR, tags_to_mark])
        for m in msgs:
            logger.debug(m)

        for child in frame.child_frames:
            if child.is_detached():
                continue
            elem = child.frame_element()
            if elem.content_frame() != child:
                continue
            sandbox = elem.get_attribute("sandbox")
            if sandbox is not None and "allow-scripts" not in sandbox.split():
                continue
            child_bid = elem.get_attribute(BID_ATTR)
            if child_bid is None:
                if lenient:
                    continue
                raise MarkingError("Cannot mark a child frame without a bid.")
            _recurse(child, child_bid)

    _recurse(page.main_frame, "")


def _unmark_elements(page):
    js = _load_js("frame_unmark_elements.js")
    for frame in page.frames:
        if _is_skip_frame(frame.url):
            continue
        try:
            if frame != page.main_frame:
                elem = frame.frame_element()
                if elem.content_frame() != frame:
                    continue
                sandbox = elem.get_attribute("sandbox")
                if sandbox is not None and "allow-scripts" not in sandbox.split():
                    continue
                if elem.get_attribute(BID_ATTR) is None:
                    continue
            frame.evaluate(js)
        except playwright.sync_api.Error as e:
            if "detached" not in str(e).lower():
                raise


# ---------------------------------------------------------------------------
# Internal: CDP extraction
# ---------------------------------------------------------------------------

def _extract_bid_from_aria(s: str) -> tuple[list, str]:
    m = _BID_REGEX.fullmatch(s)
    if not m:
        return [], s
    return list(m.groups()[:-1]), m.groups()[-1]


def _extract_dom_snapshot(page):
    cdp = page.context.new_cdp_session(page)
    snap = cdp.send("DOMSnapshot.captureSnapshot", {
        "computedStyles": [], "includeDOMRects": True, "includePaintOrder": True,
    })
    cdp.detach()

    for attr_name in ("aria-roledescription", "aria-description"):
        try:
            attr_id = snap["strings"].index(attr_name)
        except ValueError:
            continue
        seen = set()
        for doc in snap["documents"]:
            for node_attrs in doc["nodes"]["attributes"]:
                for i in range(0, len(node_attrs), 2):
                    if node_attrs[i] == attr_id:
                        vid = node_attrs[i + 1]
                        if vid not in seen:
                            _, cleaned = _extract_bid_from_aria(snap["strings"][vid])
                            snap["strings"][vid] = cleaned
                            seen.add(vid)
                        if snap["strings"][vid] == "":
                            del node_attrs[i:i + 2]
                        break
    return snap


def _extract_merged_axtree(page):
    cdp = page.context.new_cdp_session(page)

    tree = cdp.send("Page.getFrameTree")
    frame_ids = []
    queue = [tree["frameTree"]]
    while queue:
        f = queue.pop()
        queue.extend(f.get("childFrames", []))
        frame_ids.append(f["frame"]["id"])

    frame_trees = {}
    for fid in frame_ids:
        frame_trees[fid] = cdp.send("Accessibility.getFullAXTree", {"frameId": fid})

    for ax in frame_trees.values():
        for node in ax["nodes"]:
            items = []
            if "properties" in node:
                for i, p in enumerate(node["properties"]):
                    if p["name"] == "roledescription":
                        items, new = _extract_bid_from_aria(p["value"]["value"])
                        p["value"]["value"] = new
                        if not new:
                            del node["properties"][i]
                        break
            if "description" in node:
                items2, new = _extract_bid_from_aria(node["description"]["value"])
                node["description"]["value"] = new
                if not new:
                    del node["description"]
                if not items:
                    items = items2
            if items:
                node["molmoweb_id"] = items[0]

    merged = {"nodes": []}
    for ax in frame_trees.values():
        merged["nodes"].extend(ax["nodes"])
        for node in ax["nodes"]:
            if node["role"]["value"] == "Iframe":
                fid = (
                    cdp.send("DOM.describeNode", {"backendNodeId": node["backendDOMNodeId"]})
                    .get("node", {}).get("frameId")
                )
                if fid and fid in frame_trees:
                    root = frame_trees[fid]["nodes"][0]
                    node["childIds"].append(root["nodeId"])
                elif fid:
                    logger.debug(f"AXTree merging: frameId '{fid}' not in extracted trees, skipping")

    cdp.detach()
    return merged


def _extract_extra_properties(dom_snapshot) -> dict:
    strings = dom_snapshot["strings"]

    def _str(idx):
        return strings[idx] if idx != -1 else None

    def _find_str_id(s):
        try:
            return strings.index(s)
        except ValueError:
            return -1

    bid_id = _find_str_id(BID_ATTR)
    vis_id = _find_str_id(VIS_ATTR)
    som_id = _find_str_id(SOM_ATTR)

    doc_meta = {0: {"parent": None}}
    to_process = [0]

    while to_process:
        doc_idx = to_process.pop()
        doc = dom_snapshot["documents"][doc_idx]
        children = doc["nodes"]["contentDocumentIndex"]
        for node_idx, child_doc in zip(children["index"], children["value"]):
            doc_meta[child_doc] = {"parent": {"doc": doc_idx, "node": node_idx}}
            to_process.append(child_doc)

        parent = doc_meta[doc_idx]["parent"]
        abs_x = abs_y = 0.0
        if parent:
            pdoc, pnode = parent["doc"], parent["node"]
            try:
                li = dom_snapshot["documents"][pdoc]["layout"]["nodeIndex"].index(pnode)
                bounds = dom_snapshot["documents"][pdoc]["layout"]["bounds"][li]
                abs_x = doc_meta[pdoc]["abs_x"] + bounds[0]
                abs_y = doc_meta[pdoc]["abs_y"] + bounds[1]
            except ValueError:
                pass

        doc_meta[doc_idx]["abs_x"] = abs_x - doc["scrollOffsetX"]
        doc_meta[doc_idx]["abs_y"] = abs_y - doc["scrollOffsetY"]

        n_nodes = len(doc["nodes"]["parentIndex"])
        nodes = [{"bid": None, "vis": None, "bbox": None, "click": False, "som": None} for _ in range(n_nodes)]

        for ni in doc["nodes"]["isClickable"]["index"]:
            nodes[ni]["click"] = True

        for ni, attrs in enumerate(doc["nodes"]["attributes"]):
            for i in range(0, len(attrs), 2):
                nid, vid = attrs[i], attrs[i + 1]
                if nid == bid_id:
                    nodes[ni]["bid"] = _str(vid)
                elif nid == vis_id:
                    nodes[ni]["vis"] = float(_str(vid))
                elif nid == som_id:
                    nodes[ni]["som"] = _str(vid) == "1"

        for ni, bounds, cr in zip(
            doc["layout"]["nodeIndex"], doc["layout"]["bounds"], doc["layout"]["clientRects"],
        ):
            if cr:
                b = bounds.copy()
                b[0] += doc_meta[doc_idx]["abs_x"]
                b[1] += doc_meta[doc_idx]["abs_y"]
                nodes[ni]["bbox"] = b

        doc_meta[doc_idx]["nodes"] = nodes

    result = {}
    for dm in doc_meta.values():
        for n in dm.get("nodes", []):
            if n["bid"]:
                if n["bid"] in result:
                    logger.warning(f"duplicate bid={repr(n['bid'])}")
                result[n["bid"]] = {
                    "visibility": n["vis"],
                    "bbox": n["bbox"],
                    "clickable": n["click"],
                    "set_of_marks": n["som"],
                }
    return result
