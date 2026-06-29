"""
Corporate SSL inspection bypass (Netskope / self-signed cert environments).

This module MUST be imported before any library that makes HTTPS connections.
It is imported at the top of ``config/settings.py`` so that the settings
singleton applies the patch at the earliest possible moment.

Do NOT import this module more than once; Python's module cache ensures
the code runs exactly once regardless of how many times it is imported.
"""

import os

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

try:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]
except Exception:
    pass

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    import urllib3.util.ssl_

    _orig_urllib3_ctx = urllib3.util.ssl_.create_default_context  # type: ignore[attr-defined]

    def _patched_urllib3_ctx(*args, **kwargs):
        ctx = _orig_urllib3_ctx(*args, **kwargs)
        ctx.check_hostname = False
        import ssl as _ssl
        ctx.verify_mode = _ssl.CERT_NONE
        return ctx

    urllib3.util.ssl_.create_default_context = _patched_urllib3_ctx  # type: ignore[attr-defined]
except Exception:
    pass

try:
    import httpx

    _orig_httpx_client = httpx.Client.__init__

    def _patched_httpx_client(self, *args, **kwargs):
        kwargs["verify"] = False
        _orig_httpx_client(self, *args, **kwargs)

    httpx.Client.__init__ = _patched_httpx_client  # type: ignore[method-assign]

    _orig_httpx_async = httpx.AsyncClient.__init__

    def _patched_httpx_async(self, *args, **kwargs):
        kwargs["verify"] = False
        _orig_httpx_async(self, *args, **kwargs)

    httpx.AsyncClient.__init__ = _patched_httpx_async  # type: ignore[method-assign]
except Exception:
    pass

try:
    import requests

    _orig_requests_req = requests.Session.request

    def _patched_requests_req(self, method, url, *args, **kwargs):
        kwargs["verify"] = False
        return _orig_requests_req(self, method, url, *args, **kwargs)

    requests.Session.request = _patched_requests_req  # type: ignore[method-assign]
except Exception:
    pass
