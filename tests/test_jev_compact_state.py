"""Lossless SDK-state and receipt binding controls; zero external calls."""
import copy
import json
from pathlib import Path

import pytest
from typesafe_sdk import TypeSafeAPIError

from sextant import evaluators, wire
from sextant.domain import Receipt, digest
from sextant.jev_transport import JevResult
from sextant.runtime import AttemptContext, Invocation, NotSent

REQUEST=json.loads((Path(__file__).parent/'fixtures'/'jev_compact_state_public_request.json').read_text())
CONTEXT=AttemptContext('public-op','public-attempt','public-mission',1,'public-effect')


def response(questions):
    answers={}
    for key,q in questions.items():
        if q['type']=='noul':answers[key]={'type':'noul','noul':0.5}
        else:
            options=wire.declared_options(q);answers[key]={'type':q['type'],'probabilities':{o:1/len(options) for o in options}}
            if q['type']=='score':answers[key]['score']=0
    return JevResult('jev-1.13.0',answers,{'input_tokens':1,'output_tokens':1},1.0,'public-fixture-request')


def test_actual_rejected_public_request_complete_state_is_losslessly_sent_as_text(monkeypatch):
    calls=[]
    def fake(state,questions,model):
        calls.append((state,questions,model));return response(questions)
    monkeypatch.setattr(evaluators,'call_jev',fake)
    original=copy.deepcopy(REQUEST);result=evaluators.TypeSafeExecutor().invoke(original,CONTEXT)
    assert original==REQUEST and len(calls)==1
    state,questions,model=calls[0]
    assert type(state) is str and digest(json.loads(state))==digest(REQUEST['state'])
    assert questions==REQUEST['questions'] and model==REQUEST['model']
    iso=result.receipt.isolation
    assert iso['logical_state_hash']==digest(REQUEST['state'])
    assert iso['sdk_state_hash']==digest(state)
    assert iso['sdk_request_hash']==digest(iso['sdk_request'])
    assert iso['sdk_request']=={'state':state,'questions':questions,'model':model}
    assert result.receipt.request_hash==digest(REQUEST)
    assert result.receipt.output_hash==digest(result.output)
    assert 'actual HTTPS body not captured' in iso['observation_boundary']


@pytest.mark.parametrize('state',[{'x':[False,0,1,1.0,None,'中文',{'v':'\\n\n'}]},
                                  [False,1.0,None,{'x':[1]}],False,0,1.0,None])
def test_nonstring_json_state_roundtrip_preserves_every_type(state):
    encoded=evaluators.sdk_state(state)
    assert digest(json.loads(encoded))==digest(state)


def test_plain_string_is_preserved_without_new_quotes():
    text='JSON-looking text {"x":false} stays literal\n中文'
    assert evaluators.sdk_state(text)==text


@pytest.mark.parametrize('state',[{'bad':float('nan')},{'bad':float('inf')},{'bad':'\ud800'},
                                 {1:'non-JSON-key'},{'tuple':(1,2)},{'set':{1,2}}])
def test_nonjson_or_invalid_utf8_never_enters_sdk(monkeypatch,state):
    def forbidden(*args):raise AssertionError('SDK entry forbidden')
    monkeypatch.setattr(evaluators,'call_jev',forbidden)
    request={**REQUEST,'state':state}
    with pytest.raises(NotSent) as caught:evaluators.TypeSafeExecutor().invoke(request,CONTEXT)
    assert caught.value.code=='invalid_typed_state'


def test_http_window_failure_retains_code_encoding_unknown_usage_and_no_retry(monkeypatch):
    calls=[]
    def failed(state,questions,model):
        calls.append(state)
        raise TypeSafeAPIError(400,{'detail':{'error_type':'max_tokens_exceeded','private_other_field':'never-copy'}},
                               {'x-typesafe-request-id':'public-error'})
    monkeypatch.setattr(evaluators,'call_jev',failed)
    result=evaluators.TypeSafeExecutor().invoke(REQUEST,CONTEXT)
    assert len(calls)==1 and result.receipt.status=='failed' and result.receipt.usage=={}
    assert result.output['server_error_type']=='max_tokens_exceeded'
    assert result.output['server_usage']=='unknown' and 'never-copy' not in json.dumps(result.output)
    assert result.receipt.isolation['sdk_state_hash']==digest(calls[0])
    assert result.receipt.request_hash==digest(REQUEST) and result.receipt.output_hash==digest(result.output)


def test_runtime_row_keeps_logical_pack_and_actual_wrapper_receipt_binding():
    class Port:
        def capabilities(self):return evaluators.TypeSafeExecutor().capabilities()
    class Runtime:
        def dispatch(self,token,executor,request,**kwargs):
            return Invocation(Receipt(executor='typesafe',request_hash=digest(request),output_hash=digest({}),
                                      started_at='2026-10-02T00:00:00+00:00',ended_at='2026-10-02T00:00:01+00:00',
                                      status='completed',isolation={'state_encoding':evaluators.JEV_STATE_ENCODING}),
                              {'model_id':'jev-1.13.0','answers':{'q':{'type':'noul','noul':0.5}},
                               'usage':{},'latency_ms':1,'request_id':'public-fixture'})
    client=evaluators.RuntimeJevClient(Runtime(),None,[],executor=Port())
    assert client.transport=='controlled:runtime-executor'
    client._dispatched({'nested':[False,1.0]}, {'q':{'type':'noul','instructions':'Public fixture?'}},'jev-1.13.0')
    assert client.last_wire['local_sdk_parameters']['state_encoding']==evaluators.JEV_STATE_ENCODING
    assert client.last_wire['request_hash']==digest({'state':{'nested':[False,1.0]},
        'questions':{'q':{'type':'noul','instructions':'Public fixture?'}},'model':'jev-1.13.0',
        'purpose':'V1 advisory role diagnostic'})
    assert wire.request_digest(model_id='jev-1.13.0',transport=client.transport,state={'x':1},questions={}) != \
           wire.request_digest(model_id='jev-1.13.0',transport='typesafe',state={'x':1},questions={})


def test_cyclic_state_is_not_sent(monkeypatch):
    state=[];state.append(state)
    monkeypatch.setattr(evaluators,'call_jev',lambda *args:pytest.fail('SDK entry forbidden'))
    with pytest.raises(NotSent):
        evaluators.TypeSafeExecutor().invoke({**REQUEST,'state':state},CONTEXT)


def test_complete_runtime_row_binds_logical_and_exact_sdk_parameters(monkeypatch):
    calls=[]
    def fake(state,questions,model):
        calls.append((state,questions,model));return response(questions)
    monkeypatch.setattr(evaluators,'call_jev',fake)
    class Runtime:
        def dispatch(self,token,executor,request,**kwargs):
            return executor.invoke(request,CONTEXT)
    client=evaluators.RuntimeJevClient(Runtime(),None,[])
    card=client.cards['R1']
    pack={field:{'content':[False,1.0,None,'Public literal']} for field in card.state_pack}
    rendered=client.render(card,pack)
    _,row=client._call(card,pack,rendered,variant='base',role='primary',run_id='public-controlled-run',input_language='en')
    assert len(calls)==1 and len(client.rows)==1
    assert row['transport']=='typesafe-canonical-json-text/1' and row['request']['state']==pack
    assert wire.payload_problems(row)==[]
    iso=row['runtime_dispatch']['local_sdk_parameters']
    assert json.loads(iso['sdk_request']['state'])==pack
    assert iso['sdk_request_hash']==digest(iso['sdk_request'])
    assert iso['sdk_request']['state']==calls[0][0]
    assert iso['sdk_request']['questions']==row['request']['questions']
    binding=row['runtime_dispatch']['diagnostic_binding']
    assert binding=={'card_id':'R1','run_id':'public-controlled-run','variant':'base','call_role':'primary',
                    'sdk_state_sha256':digest(calls[0][0]),'sdk_questions_sha256':digest(calls[0][1])}
    assert set(iso['sdk_request'])=={'state','questions','model'}
    assert client._diagnostic_binding is None


def test_falsy_explicit_executor_is_retained_and_never_selects_default_port(monkeypatch):
    class FalsyPort:
        name='controlled:public-fixture'
        def __bool__(self):return False
        def capabilities(self):return evaluators.TypeSafeExecutor().capabilities()
        def invoke(self,request,context):raise AssertionError('Only selection is tested')
        def cancel(self,context):return {'status':'unsupported'}
        def reconcile(self,context):return None
    monkeypatch.setattr(evaluators,'call_jev',lambda *args:pytest.fail('real SDK entry forbidden'))
    port=FalsyPort();client=evaluators.RuntimeJevClient(None,None,[],executor=port)
    assert client.executor is port and client.transport=='controlled:runtime-executor'


@pytest.mark.parametrize('metadata',[{'purpose':'\ud800'},{'purpose':float('nan')},{'extra':b'not JSON'}])
def test_invalid_complete_request_metadata_is_refused_before_wrapper(monkeypatch,metadata):
    monkeypatch.setattr(evaluators,'call_jev',lambda *args:pytest.fail('SDK entry forbidden'))
    with pytest.raises(NotSent) as caught:
        evaluators.TypeSafeExecutor().invoke({**REQUEST,**metadata},CONTEXT)
    assert caught.value.code=='invalid_typed_state'
