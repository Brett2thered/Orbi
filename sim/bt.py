import sys, json
import measure as M
kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
M.bounce(None, kw, do_course=len(sys.argv) > 2)
