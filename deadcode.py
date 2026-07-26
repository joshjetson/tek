"""Find module-level names that nothing outside their own file references."""
import ast
import os
import re

HOME = "/home/super"
FILES = [f for f in sorted(os.listdir(HOME)) if f.endswith(".py")]
for d in ("tools", "tests"):
    FILES += [os.path.join(d, f) for f in sorted(os.listdir(os.path.join(HOME, d)))
              if f.endswith(".py")]

RUNTIME = ["tekvector.py", "tekfb.py", "tekhead.py", "tekfdl.py",
           "tekrig.py", "tekcam.py", "tekrun.py"]

src = {}
for f in FILES:
    try:
        src[f] = open(os.path.join(HOME, f), errors="ignore").read()
    except OSError:
        pass

print("%-14s %-26s %s" % ("module", "top-level name", "referenced elsewhere?"))
print("-" * 68)
total_dead = 0
dead_lines = 0
for f in RUNTIME:
    try:
        tree = ast.parse(src[f])
    except SyntaxError:
        continue
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.append((node.name, node.lineno,
                          getattr(node, "end_lineno", node.lineno)))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("__"):
                    names.append((t.id, node.lineno,
                                  getattr(node, "end_lineno", node.lineno)))
    others = {g: s for g, s in src.items() if g != f}
    for name, lo, hi in names:
        if name.startswith("_") and name not in ("_march", "_ridge", "_blob"):
            used_self = len(re.findall(r"\b%s\b" % re.escape(name), src[f])) > 1
            used_out = any(re.search(r"\b%s\b" % re.escape(name), s)
                           for s in others.values())
            if used_self or used_out:
                continue
        hits = [g for g, s in others.items()
                if re.search(r"\b%s\b" % re.escape(name), s)]
        if not hits:
            n = max(1, hi - lo + 1)
            # is it used inside its own module?
            self_uses = len(re.findall(r"\b%s\b" % re.escape(name), src[f]))
            tag = "DEAD (%d lines)" % n if self_uses <= 1 else "internal only"
            if self_uses <= 1:
                total_dead += 1
                dead_lines += n
            print("%-14s %-26s %s" % (f, name, tag))
print("-" * 68)
print("%d unreferenced top-level names, ~%d lines" % (total_dead, dead_lines))
