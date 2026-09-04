"""Render README.md from README.tmpl.md, filling {{path.to.key:fmt}} from two results files.

S10 runs two models (the proxy transformer on EC2, nanoGPT on Colab) so both result files
are loaded and namespaced under "proxy." and "nanogpt." — a placeholder like
{{proxy.1a_shapes.tokens}} reads results_proxy.json, {{nanogpt.5_mfu.mfu_pct:.1f}} reads
results_nanogpt.json. A placeholder that does not resolve is an error, not a silently-empty
span: no number in the write-up is typed by hand, every one is looked up from a file a
notebook actually wrote on its last run.
"""
import json, pathlib, re, sys

results = {}
for name in ("proxy", "nanogpt"):
    p = pathlib.Path(f"results_{name}.json")
    if p.exists():
        results[name] = json.loads(p.read_text())

tmpl = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "README.tmpl.md").read_text()

PLACEHOLDER = re.compile(r"\{\{([A-Za-z0-9_.\[\]]+)(?::([^}]+))?\}\}")
missing = []


def lookup(path):
    node = results
    for part in path.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        else:
            node = node[part]
    return node


def render(m):
    path, fmt = m.group(1), m.group(2)
    try:
        val = lookup(path)
    except (KeyError, IndexError, ValueError, TypeError):
        missing.append(f"{path}  (no such key in results_proxy.json / results_nanogpt.json)")
        return f"<<MISSING {path}>>"
    if not fmt:
        return str(val)
    try:
        return format(val, fmt)
    except (ValueError, TypeError) as e:
        missing.append(f"{path}  (bad format {fmt!r} for {type(val).__name__}: {e})")
        return f"<<BADFMT {path}>>"


out = PLACEHOLDER.sub(render, tmpl)
if missing:
    print("unresolved placeholders:", *sorted(set(missing)), sep="\n  ")
    sys.exit(1)

pathlib.Path("README.md").write_text(out)
print(f"README.md: {len(out.splitlines())} lines, "
      f"{len(PLACEHOLDER.findall(tmpl))} values filled from "
      f"{', '.join(f'results_{n}.json' for n in results)}")
