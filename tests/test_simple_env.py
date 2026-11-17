import unittest
from unittest import mock
import sys

# Mock external runtime dependencies for test isolation
sys.modules["playwright"] = mock.MagicMock()
sys.modules["playwright.sync_api"] = mock.MagicMock()
sys.modules["agent.actions"] = mock.MagicMock()
sys.modules["utils.envs.action_executor"] = mock.MagicMock()
sys.modules["utils.axtree"] = mock.MagicMock()

from utils.envs.browser_env import SimpleEnv


class TestSimpleEnvParams(unittest.TestCase):
    def test_default_params_preserves_original_behavior(self):
        mock_pw = mock.MagicMock()
        mock_browser = mock.MagicMock()
        mock_context = mock.MagicMock()
        mock_page = mock.MagicMock()

        mock_pw.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page

        with mock.patch("utils.envs.browser_env._start_playwright", return_value=mock_pw):
            env = SimpleEnv()
            env._launch()

        mock_pw.chromium.launch.assert_called_once_with(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        mock_browser.new_context.assert_called_once_with(
            viewport={"width": 1280, "height": 720},
        )
        self.assertEqual(env._get_info(), {})

    def test_custom_locale_timezone_geolocation_and_proxy(self):
        mock_pw = mock.MagicMock()
        mock_browser = mock.MagicMock()
        mock_context = mock.MagicMock()
        mock_page = mock.MagicMock()

        mock_pw.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page

        with mock.patch("utils.envs.browser_env._start_playwright", return_value=mock_pw):
            env = SimpleEnv(
                locale="en-US",
                timezone_id="America/New_York",
                geolocation={"latitude": 40.7128, "longitude": -74.0060},
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                proxy={"server": "http://127.0.0.1:8080"},
            )
            env._launch()

        mock_pw.chromium.launch.assert_called_once_with(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
            proxy={"server": "http://127.0.0.1:8080"},
        )
        mock_browser.new_context.assert_called_once_with(
            viewport={"width": 1280, "height": 720},
            locale="en-US",
            timezone_id="America/New_York",
            geolocation={"latitude": 40.7128, "longitude": -74.0060},
            permissions=["geolocation"],
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            proxy={"server": "http://127.0.0.1:8080"},
        )
        info = env._get_info()
        self.assertEqual(info["locale"], "en-US")
        self.assertEqual(info["timezone_id"], "America/New_York")
        self.assertEqual(info["proxy"], "http://127.0.0.1:8080")


if __name__ == "__main__":
    unittest.main()
