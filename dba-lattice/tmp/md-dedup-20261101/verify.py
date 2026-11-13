from pathlib import Path
import difflib,hashlib,json,re
from urllib.parse import unquote
root=Path(__file__).resolve().parents[2]
work=Path(__file__).resolve().parent
initial=json.loads((work/'inventory.before.json').read_text())
backups=work/'before'
active={str(p.relative_to(root)):p for p in root.rglob('*.md')
        if not {'tmp','results','__pycache__'} & set(p.relative_to(root).parts)}
assert set(active)==set(initial),'active Markdown inventory changed unexpectedly'
changed=[]
new_links=[]
before_bytes=0
after_bytes=0
diffs=[]
for rel,p in sorted(active.items()):
    backup=backups/(rel+'.before')
    if not backup.exists():
        assert hashlib.sha256(p.read_bytes()).hexdigest()==initial[rel]['sha256'],rel
        continue
    original=backup.read_text()
    current=p.read_text()
    if current==original:continue
    changed.append(rel)
    before_bytes+=len(original.encode())
    after_bytes+=len(current.encode())
    diffs.extend(difflib.unified_diff(original.splitlines(True),current.splitlines(True),fromfile=rel+':before',tofile=rel))
    assert sum(bool(re.match(r'^\s*```',line)) for line in current.splitlines())%2==0,rel
    links=lambda text:set(re.findall(r'\]\(([^)]+)\)',text))
    for target in sorted(links(current)-links(original)):
        if re.match(r'^\w+://',target):continue
        local=unquote(target.split('#')[0])
        assert (p.parent/local).exists(),(rel,target)
        new_links.append(dict(document=rel,target=target))
preissues=json.loads((work/'manifest-issues.before.json').read_text())
postissues={}
verified=[]
for manifest in root.glob('*/CHECKSUMS.sha256'):
    issues=[]
    count=0
    for line in manifest.read_text().splitlines():
        if not line.strip():continue
        digest,name=line.split(maxsplit=1)
        p=manifest.parent/name
        if not p.exists():issues.append(dict(path=name,status='missing'))
        elif hashlib.sha256(p.read_bytes()).hexdigest()!=digest:issues.append(dict(path=name,status='mismatch'))
        else:count+=1
    rel=str(manifest.relative_to(root))
    assert {(a['path'],a['status']) for a in issues}<={(a['path'],a['status']) for a in preissues[rel]},rel
    postissues[rel]=issues
    if not issues:verified.append(dict(manifest=rel,verified_entries=count))
    for changed_rel in changed:
        p=root/changed_rel
        if p.parent==manifest.parent:
            assert not any(a['path'].removeprefix('./')==p.name for a in issues),(rel,p.name)
tex_hash=hashlib.sha256((root/'TeX/main.tex').read_bytes()).hexdigest()
previous=json.loads((root/'tmp/figures-20261101/checks.json').read_text())
assert tex_hash==previous['source_sha256'],'LaTeX changed during Markdown cleanup'
checks=dict(active_markdown_reviewed=len(active),markdown_changed=len(changed),changed_files=changed,
    unchanged_files=sorted(set(active)-set(changed)),bytes_before=before_bytes,bytes_after=after_bytes,
    bytes_removed=before_bytes-after_bytes,new_relative_links_checked=new_links,
    balanced_code_fences=True,latex_unchanged=True,portable_manifests_verified=verified,
    historical_manifest_issues_remaining={k:len(v) for k,v in postissues.items()},
    code_changes=False,simulations_run=False)
(work/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
(work/'manifest-issues.after.json').write_text(json.dumps(postissues,indent=2)+'\n')
(work/'documentation.diff').write_text(''.join(diffs))
print(json.dumps({k:v for k,v in checks.items() if k not in ['unchanged_files','new_relative_links_checked']},indent=2))
