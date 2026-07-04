import pathlib
root = pathlib.Path('E:/projects/laecon')
files = {
    'spec/constitution.md': root / 'spec/constitution.md',
    'spec/todo.md': root / 'spec/todo.md',
    'spec/adr/_template.md': root / 'spec/adr/_template.md',
    'spec/adr/README.md': root / 'spec/adr/README.md',
    'spec/harness/_template.md': root / 'spec/harness/_template.md',
    'spec/specs/000-bootstrap/spec.md': root / 'spec/specs/000-bootstrap/spec.md',
    'contract.md': root / 'contract.md',
    'README.md': root / 'README.md',
}
for name, p in files.items():
    if p.exists():
        text = p.read_text('utf-8')
        print(f'  {len(text):>6} chars  {p.stat().st_size:>6} bytes  {name}')
    else:
        print(f'  MISSING  {name}')
ot = root / 'spec/opencode-templates'
templates = sorted(ot.iterdir()) if ot.exists() else []
print(f'\n  {len(templates)} files in spec/opencode-templates/')
for t in templates:
    text = t.read_text('utf-8')
    print(f'  {len(text):>6} chars  {t.stat().st_size:>6} bytes  opencode-templates/{t.name}')
print(f'\n  spec/design-direction.md exists: {(root / "spec/design-direction.md").exists()}')
print(f'  CONSTITUTION.md: {(root / "CONSTITUTION.md").stat().st_size} bytes (should be original 926-line)')