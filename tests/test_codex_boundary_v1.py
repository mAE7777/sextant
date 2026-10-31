"""Package pins cannot be replaced by caller flags or cached success receipts."""
import hashlib
import json

import pytest

from sextant.providers import codex_boundary as boundary


@pytest.fixture
def bound_package(tmp_path, monkeypatch):
    package = tmp_path / 'sextant'
    (package / 'providers').mkdir(parents=True)
    (package / 'resources').mkdir()
    monkeypatch.setattr(boundary, '__file__', str(package / 'providers/codex_boundary.py'))
    monkeypatch.setattr(boundary.platform, 'system', lambda: 'Darwin')
    names = ['providers/codex_native.py', 'providers/codex_transport.py', 'isolation.py', 'domain.py', 'jev_transport.py']
    source = {}
    for name in names:
        p = package / name
        p.write_text('controlled public source ' + name)
        source[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    binary,host = tmp_path/'client',tmp_path/'host'
    binary.write_bytes(b'public native bytes');host.write_bytes(b'public host bytes')
    record = {'schema_version':1,'scope':'configured-client-pure-proposals',
              'auth_route':'native-codex-chatgpt-subscription',
              'pairs':[{'model':'gpt-6.1-sol','effort':'xhigh'}], 'source_hashes':source,
              'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
              'host_sha256':hashlib.sha256(host.read_bytes()).hexdigest(),
              'evidence':{'public_test':'a'*64,'independent_outcome':'b'*64},
              'trust_assumptions':['public fixture only, not real qualification']}
    p=package/'resources/native-codex-qualification.json';p.write_text(json.dumps(record))
    monkeypatch.setattr(boundary,'QUALIFICATION_SHA256',hashlib.sha256(p.read_bytes()).hexdigest())
    return package,binary,host,p


def test_same_bytes_at_a_new_installation_path_can_qualify_fixture(bound_package):
    _,binary,host,_=bound_package
    result=boundary.boundary_status(binary,host,'gpt-6.1-sol','xhigh')
    assert result['qualified'] is True and result['backend_identity']=='unknown'


@pytest.mark.parametrize('change',['source','client','host','record','model','effort','platform'])
def test_changed_boundary_bytes_or_scope_cannot_use_old_qualification(bound_package,monkeypatch,change):
    package,binary,host,record=bound_package
    model,effort='gpt-6.1-sol','xhigh'
    if change=='source':(package/'providers/codex_transport.py').write_text('different code')
    elif change=='client':binary.write_bytes(b'new client')
    elif change=='host':host.write_bytes(b'new host')
    elif change=='record':record.write_text(record.read_text()+' ')
    elif change=='model':model='gpt-6-astra'
    elif change=='effort':effort='max'
    else:monkeypatch.setattr(boundary.platform,'system',lambda:'Linux')
    monkeypatch.setenv('SEXTANT_CODEX_BOUNDARY_QUALIFIED','true')
    assert boundary.boundary_status(binary,host,model,effort)['qualified'] is False
