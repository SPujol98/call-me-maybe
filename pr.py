Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/spujol-s/42/cursus/CallMeMaybe/src/__main__.py", line 71, in <module>
    main()
    ~~~~^^
  File "/home/spujol-s/42/cursus/CallMeMaybe/src/__main__.py", line 60, in main
    result_json = json.loads(decoded)
  File "/home/spujol-s/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/json/__init__.py", line 346, in loads
    return _default_decoder.decode(s)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "/home/spujol-s/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/json/decoder.py", line 345, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
               ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/spujol-s/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/json/decoder.py", line 361, in raw_decode
    obj, end = self.scan_once(s, idx)
               ~~~~~~~~~~~~~~^^^^^^^^
json.decoder.JSONDecodeError: Invalid control character at: line 1 column 180 (char 179)
(call-me-maybe) ➜  CallMeMaybe git:(master) ✗ 