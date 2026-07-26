import ast
import os
import subprocess

HOME = "/home/super"
mine = []
for f in sorted(os.listdir(HOME)):
    p = os.path.join(HOME, f)
    if os.path.isfile(p) and f.endswith((".py", ".sh", ".md")):
        mine.append(f)

# who imports whom
mods = {f[:-3] for f in mine if f.endswith(".py")}
imports = {}
for f in mine:
    if not f.endswith(".py"):
        continue
    try:
        tree = ast.parse(open(os.path.join(HOME, f)).read())
    except SyntaxError:
        imports[f] = set(["<syntax error>"])
        continue
    got = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in mods:
                    got.add(a.name)
        elif isinstance(node, ast.ImportFrom) and node.module in mods:
            got.add(node.module)
    imports[f] = got

used_by = {m: [] for m in mods}
for f, deps in imports.items():
    for d in deps:
        used_by[d].append(f[:-3])

print("%-26s %6s  %s" % ("file", "lines", "imported by"))
for f in mine:
    p = os.path.join(HOME, f)
    n = sum(1 for _ in open(p, errors="ignore"))
    m = f[:-3]
    who = ", ".join(sorted(used_by.get(m, []))) if f.endswith(".py") else ""
    print("%-26s %6d  %s" % (f, n, who or ("-" if f.endswith(".py") else "")))

print("\nservice ExecStart:")
print("  " + subprocess.run(
    ["systemctl", "show", "tek-display", "-p", "ExecStart", "--value"],
    capture_output=True, text=True).stdout.strip()[:160])
print("\ndirs:", [d for d in sorted(os.listdir(HOME))
                 if os.path.isdir(os.path.join(HOME, d)) and not d.startswith(".")])
