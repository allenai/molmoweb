/**
 * Go through all DOM elements in the frame (including shadowDOMs), give them unique
 * identifiers (bid), and store custom data in ARIA attributes.
 */
async ([parent_bid, bid_attr_name, tags_to_mark]) => {

    // Plain objects for tag lookup — immune to page-level Set polyfill overrides
    // https://www.w3schools.com/tags/
    const html_tags = {
        "a":1,"abbr":1,"acronym":1,"address":1,"applet":1,"area":1,"article":1,"aside":1,"audio":1,
        "b":1,"base":1,"basefont":1,"bdi":1,"bdo":1,"big":1,"blockquote":1,"body":1,"br":1,"button":1,
        "canvas":1,"caption":1,"center":1,"cite":1,"code":1,"col":1,"colgroup":1,"data":1,"datalist":1,
        "dd":1,"del":1,"details":1,"dfn":1,"dialog":1,"dir":1,"div":1,"dl":1,"dt":1,"em":1,"embed":1,
        "fieldset":1,"figcaption":1,"figure":1,"font":1,"footer":1,"form":1,"frame":1,"frameset":1,
        "h1":1,"h2":1,"h3":1,"h4":1,"h5":1,"h6":1,"head":1,"header":1,"hgroup":1,"hr":1,"html":1,"i":1,
        "iframe":1,"img":1,"input":1,"ins":1,"kbd":1,"label":1,"legend":1,"li":1,"link":1,"main":1,
        "map":1,"mark":1,"menu":1,"meta":1,"meter":1,"nav":1,"noframes":1,"noscript":1,"object":1,
        "ol":1,"optgroup":1,"option":1,"output":1,"p":1,"param":1,"picture":1,"pre":1,"progress":1,
        "q":1,"rp":1,"rt":1,"ruby":1,"s":1,"samp":1,"script":1,"search":1,"section":1,"select":1,
        "small":1,"source":1,"span":1,"strike":1,"strong":1,"style":1,"sub":1,"summary":1,"sup":1,
        "svg":1,"table":1,"tbody":1,"td":1,"template":1,"textarea":1,"tfoot":1,"th":1,"thead":1,
        "time":1,"title":1,"tr":1,"track":1,"tt":1,"u":1,"ul":1,"var":1,"video":1,"wbr":1
    };
    const set_of_marks_tags = {
        "input":1,"textarea":1,"select":1,"button":1,"a":1,"iframe":1,"video":1,"li":1,"td":1,"option":1
    };

    let molmoweb_first_visit = false;
    // if no yet set, set the frame (local) element counter to 0
    if (!("molmoweb_elem_counter" in window)) {
        window.molmoweb_elem_counter = 0;
        window.molmoweb_frame_id_generator = new IFrameIdGenerator();
        molmoweb_first_visit = true;
    }
    // Track elements awaiting their first intersection observer visit via a counter + attribute.
    let _mw_remaining = 0;
    let intersection_observer = new IntersectionObserver(
        entries => {
          entries.forEach(entry => {
            let elem = entry.target;
            elem.setAttribute('molmoweb_visibility_ratio', Math.round(entry.intersectionRatio * 100) / 100);
            if (elem.getAttribute('_mw_awaiting') === '1') {
                elem.setAttribute('_mw_awaiting', '0');
                _mw_remaining--;
            }
          })
        },
        {
            threshold: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }
    )

    let all_bids = {};

    // get all DOM elements in the current frame (does not include elements in shadowDOMs)
    let elements = Array.from(document.querySelectorAll('*'));
    let som_buttons = [];
    let i = 0;
    while (i < elements.length) {
        const elem = elements[i];
        // add shadowDOM elements to the elements array, in such a way that order is preserved
        // TODO: do we really need the order preserved?
        if (elem.shadowRoot !== null) {
            elements = new Array(
                ...Array.prototype.slice.call(elements, 0, i + 1),
                ...Array.from(elem.shadowRoot.querySelectorAll("*")),
                ...Array.prototype.slice.call(elements, i + 1)
            );
        }
        i++;
        // decide if the current element should be marked or not
        switch (tags_to_mark) {
            // mark all elements
            case "all":
                break;
            // mark only standard HTML tags
            case "standard_html":
                if (!elem.tagName || !(elem.tagName.toLowerCase() in html_tags)) {
                    // continue the loop, i.e., move on to the next element
                    continue;
                }
                break;
            // non-recognized argument
            default:
                throw new Error(`Invalid value for parameter \"tags_to_mark\": ${JSON.stringify(tags_to_mark)}`);
        }
        // Processing element
        // register intersection callback on element, and keep track of element for waiting later
        elem.setAttribute('molmoweb_visibility_ratio', 0);
        elem.setAttribute('_mw_awaiting', '1');
        _mw_remaining++;
        intersection_observer.observe(elem);
        // write dynamic element values to the DOM
        if (typeof elem.value !== 'undefined') {
            elem.setAttribute("value", elem.value);
        }
        // write dynamic checked properties to the DOM
        if (typeof elem.checked !== 'undefined') {
            if (elem.checked === true) {
                elem.setAttribute("checked", "");
            }
            else {
                elem.removeAttribute("checked");
            }
        }
        // https://playwright.dev/docs/locators#locate-by-test-id
        // recover the element id if it has one already, else compute a new element id
        let elem_global_bid = null;
        if (elem.hasAttribute(bid_attr_name)) {
            // throw an error if the attribute is already set while this is the first visit of the page
            if (molmoweb_first_visit) {
                throw new Error(`Attribute ${bid_attr_name} already used in element ${elem.outerHTML}`);
            }
            elem_global_bid = elem.getAttribute(bid_attr_name);
            // if the bid has already been encountered, then this is a duplicate and a new bid should be set
            if (elem_global_bid in all_bids) {
                console.log(`MolmoWeb: duplicate bid ${elem_global_bid} detected, generating a new one`);
                elem_global_bid = null;
            }
        }
        if (elem_global_bid === null) {
            let elem_local_id = null;
            // iFrames get alphabetical ids: 'a', 'b', ..., 'z', 'aA', 'aB' etc.
            if (['iframe', 'frame'].includes(elem.tagName.toLowerCase())) {
                elem_local_id = `${window.molmoweb_frame_id_generator.next()}`;
            }
            // other elements get numerical ids: '0', '1', '2', ...
            else {
                elem_local_id = `${window.molmoweb_elem_counter++}`;
            }
            if (parent_bid == "") {
                elem_global_bid = `${elem_local_id}`;
            }
            else {
                elem_global_bid = `${parent_bid}${elem_local_id}`;
            }
            elem.setAttribute(bid_attr_name, `${elem_global_bid}`);
        }
        all_bids[elem_global_bid] = 1;

        // Hack: store custom data inside ARIA attributes (will be available in DOM and AXTree)
        //  - elem_global_bid: global element identifier (unique over multiple frames)
        // TODO: add more data if needed (x, y coordinates, bounding box, is_visible, is_clickable etc.)
        push_bid_to_attribute(elem_global_bid, elem, "aria-roledescription");
        push_bid_to_attribute(elem_global_bid, elem, "aria-description");  // fallback for generic nodes

        // set-of-marks flag (He et al. 2024)
        // https://github.com/MinorJerry/WebVoyager/blob/main/utils.py
        elem.setAttribute("molmoweb_set_of_marks", "0");
        // click at center activates self or a child
        if (["self", "child"].includes(whoCapturesCenterClick(elem))) {
            // has valid tag name, or has click event, or triggers a pointer cursor
            if ((elem.tagName.toLowerCase() in set_of_marks_tags) || (elem.onclick != null) || (window.getComputedStyle(elem).cursor == "pointer")) {
                let rect = elem.getBoundingClientRect();
                let area = (rect.right - rect.left) * (rect.bottom - rect.top);
                // area is large enough
                if (area >= 20) {
                    // is not a child of a button (role, type, tag) set to be marked
                    if (som_buttons.every(button => !button.contains(elem))) {
                        // is not the sole child of span that has a role and is set to be marked
                        let parent = elem.parentElement;
                        if (!(parent && parent.tagName.toLowerCase() == "span" && parent.children.length === 1 && parent.getAttribute("role") && parent.getAttribute("molmoweb_set_of_marks") === "1")) {
                            // all checks have passed, flag the element for inclusion in set-of-marks
                            elem.setAttribute("molmoweb_set_of_marks", "1");
                            if (elem.matches('button, a, input[type="button"], div[role="button"]')) {
                                som_buttons.push(elem)
                            }
                            // lastly, remove the set-of-marks flag from all parents, if any
                            while (parent) {
                                if (parent.getAttribute("molmoweb_set_of_marks") === "1") {
                                    parent.setAttribute("molmoweb_set_of_marks", "0")
                                }
                                parent = parent.parentElement;
                            }
                        }
                    }
                }
            }
        }
    }

    let warning_msgs = new Array();

    // wait for all elements to be visited for visibility
    let visibility_marking_timeout = 1000;  // ms
    try {
        await until(() => _mw_remaining == 0, visibility_marking_timeout);
    } catch {
        warning_msgs.push(`Frame marking: not all elements have been visited by the intersection_observer after ${visibility_marking_timeout} ms`);
    }
    // disconnect intersection observer
    intersection_observer.disconnect();

    return warning_msgs;
}

async function until(f, timeout, interval=40) {
    return new Promise((resolve, reject) => {
        const start_time = Date.now();
        // immediate check
        if (f()) {
            resolve();
        }
        // loop check
        const wait = setInterval(() => {
            if (f()) {
                clearInterval(wait);
                resolve();
            } else if (Date.now() - start_time > timeout) {
                clearInterval(wait);
                reject();
            }
        }, interval);
    });
}


function whoCapturesCenterClick(element){
    var rect = element.getBoundingClientRect();
    var x = (rect.left + rect.right) / 2 ;
    var y = (rect.top + rect.bottom) / 2 ;
    var element_at_center = elementFromPoint(x, y); // return the element in the foreground at position (x,y)
    if (!element_at_center) {
        return "nobody";
    } else if (element_at_center === element) {
        return "self";
    } else if (element.contains(element_at_center)) {
        return "child";
    } else {
        return "non-descendant";
    }
}

function push_bid_to_attribute(bid, elem, attr){
    let original_content = "";
    if (elem.hasAttribute(attr)) {
        original_content = elem.getAttribute(attr);
    }
    let new_content = `molmoweb_id_${bid} ${original_content}`
    elem.setAttribute(attr, new_content);
}

function elementFromPoint(x, y) {
    let dom = document;
    let last_elem = null;
    let elem = null;

    do {
        last_elem = elem;
        elem = dom.elementFromPoint(x, y);
        dom = elem?.shadowRoot;
    } while(dom && elem !== last_elem);

    return elem;
}

// https://stackoverflow.com/questions/12504042/what-is-a-method-that-can-be-used-to-increment-letters#answer-12504061
class IFrameIdGenerator {
    constructor(chars = 'abcdefghijklmnopqrstuvwxyz') {
      this._chars = chars;
      this._nextId = [0];
    }

    next() {
      const r = [];
      for (let i = 0; i < this._nextId.length; i++) {
        let char = this._chars[this._nextId[i]];
        // all but first character must be upper-cased (a, aA, bCD)
        if (i < this._nextId.length - 1) {
            char = char.toUpperCase();
        }
        r.unshift(char);
      }
      this._increment();
      return r.join('');
    }

    _increment() {
      for (let i = 0; i < this._nextId.length; i++) {
        const val = ++this._nextId[i];
        if (val < this._chars.length) {
          return;
        }
        this._nextId[i] = 0;
      }
      this._nextId.push(0);
    }

    *[Symbol.iterator]() {
      while (true) {
        yield this.next();
      }
    }
  }
