"""Current grants must reach every actual planning stage; zero live calls."""
import json

import pytest
from test_application_v1 import OWNER, ControlledGenerator, call

from sextant.application import Application
from sextant.storage import Store


class ControlledCodex(ControlledGenerator):
    name = "codex"


@pytest.mark.parametrize("tools,read_paths,expected", [
    (["codex", "typesafe"], [], []),
    (["codex", "sandbox.python"], [], ["sandbox.python"]),
    (["codex", "source.file"], [], []),
    (["codex", "source.file"], ["/public/host-approved.txt"], ["source.file"]),
    (["codex", "environment"], [], ["environment"]),
])
def test_real_compilation_stages_receive_current_grant_not_unconditional_local_adapters(tmp_path,tools,read_paths,expected):
    generator = ControlledCodex()
    app = Application(Store(tmp_path / "runtime"), OWNER, executors={"codex":generator})
    start = call(app,"start",mission={"goal":"Compare candidate mechanisms within the host grant"},
                 options={"generator":"codex","jev_density":"none","environment":{"episode_id":"public-test","actions":{"experiment":{}}}},
                 authorization={"tools":tools,"send_to":{t:["*"] for t in tools},"read_paths":read_paths})
    result = call(app,"run",start,steps=1)
    assert not result.errors, result
    assert [r["purpose"] for r in generator.requests] == ["primary","challenger","synthesis"]
    for request in generator.requests:
        content = json.loads(request["prompt"])
        access = content["execution_access"]
        assert access["authorized_tools"] == sorted(tools)
        assert access["experimental_adapters"] == expected
        assert access["world_effects_authorized"] is False
        assert set(content["local_adapters"]) == set(expected) - {"environment"}
        assert "dispatcher rechecks" in access["scope"]
        assert "do not simulate real observations" in access["instruction"]
        assert "host-approved.txt" not in request["prompt"]


def test_advisory_roles_see_the_same_current_grant_without_new_authority(tmp_path,monkeypatch):
    from sextant import compiler
    observed=[]
    class AdvisoryOnly(compiler.RuntimeJevClient):
        def diagnose(self,card,pack,**kwargs):
            observed.append(pack['access'])
            return {'card_id':card,'run_id':'public-scripted','authority':'advisory',
                    'automatic_decisions':0,'derived':{},'purpose':'public fixture','uncertainty_kind':'fixture'}
    monkeypatch.setattr(compiler,'RuntimeJevClient',AdvisoryOnly)
    generator=ControlledCodex()
    app=Application(Store(tmp_path/'runtime'),OWNER,executors={'codex':generator})
    start=call(app,'start',mission={'goal':'Compare within a generation and advisory-only grant'},
               options={'generator':'codex','jev_density':'multi_level'},
               authorization={'tools':['codex','typesafe'],'send_to':{'codex':['*'],'typesafe':['*']}})
    result=call(app,'run',start,steps=1)
    assert not result.errors, result
    assert len(observed)==10 and len(generator.requests)==4
    for access in observed:
        assert access['can_verify_world'] is False
        assert access['execution_access']['authorized_tools']==['codex','typesafe']
        assert access['execution_access']['experimental_adapters']==[]
