"""Execute model-proposed target confirmations; never search or navigate on observe."""
import hashlib
import json
import re
import time
from uuid import uuid4

from .errors import RpaIssue


SEMESTER = re.compile(r'^\d{4}\s*[-—–/]\s*\d{4}\s*第[一二三123]学期$')
NAVIGATION = {'个人空间','我的课程','课程','作业','首页','门户','更多','返回'}


def proof(page, item, name):
    data = [page.url, item['frame_id'], item.get('record_id') or item.get('context_id'), name, item['context']]
    return hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()[:24]


def contains_name(item, name):
    return name == item['name'] or name in item.get('context_labels', [])


class TargetConfirmation:
    def __init__(self, task):
        self.task = task

    async def propose(self, action):
        task, adapter = self.task, self.task.adapter
        kind, name = action['kind'], action['actual_name'].strip()
        if kind not in {'course','assignment'} or not name or len(name)>255 or name in NAVIGATION:
            raise ValueError('候选类型或名称无效')
        if (task.state.get('export_attempted') or adapter.assignment_confirmed or
                (kind == 'course' and adapter.course_confirmed) or
                (kind == 'assignment' and not adapter.course_confirmed)):
            raise ValueError('当前阶段不能更改课程或作业目标')
        entry = adapter.refs.get(action['element_ref'])
        if not entry or entry[2].get('disabled') or not contains_name(entry[2], name):
            raise RpaIssue('TARGET_EVIDENCE_MISSING','候选名称必须来自当前观察中的名称或同一条记录的 context_labels')
        page, _, item = entry
        term = action['term'].strip()
        if term and not (len(term)<=255 and any(
                e['name'] == term or e.get('selected_option') == term or term in e.get('context_labels', [])
                for e in adapter.last_view.get('elements', []))):
            raise RpaIssue('SEMESTER_EVIDENCE_MISSING','学期名称必须来自当前页面证据')
        candidate = {'id':proof(page,item,name), 'name':name, 'context':item['context'][:220]}
        if adapter.approved_targets.get(kind) == candidate:
            return  # A repeated model proposal cannot ask the teacher twice.
        task.state.setdefault('requested_target', dict(task.state['target']))
        label = '课程' if kind == 'course' else '作业'
        message = f'输入的{label}为“{task.state["target"][kind+"_name"]}”，页面找到“{name}”，请确认是否为本次目标'
        task.update(target_resolution={'id':str(uuid4()),'kind':kind,'requested_name':task.state['target'][kind+'_name'],
            'term':term,'previous_term':False,'semester_fallback':False,'candidates':[candidate],
            'message':message,'model_reason':action['reason'][:300]})
        task.wait_human('TARGET_SUGGESTION',message)

    async def resolve(self, payload):
        task, adapter = self.task, self.task.adapter
        pending = task.state.get('target_resolution')
        if (not isinstance(payload,dict) or not pending or task.state['status']!='WAITING_USER'
                or task.state['control_mode']!='HUMAN' or task.state.get('export_attempted')
                or payload.get('resolution_id')!=pending['id']):
            raise ValueError('确认已过期，请刷新待处理事项')
        if not adapter or not await adapter.authenticated():
            raise ValueError('请先在浏览器中完成登录并返回教学平台')
        kind = pending['kind']
        if payload.get('choice') == 'edit':
            name = payload.get('name')
            if not isinstance(name,str) or not name.strip() or len(name.strip())>255:
                raise ValueError('请填写正确的课程或作业名称')
            task.update(target={**task.state['target'],kind+'_name':name.strip()},target_resolution=None)
            adapter.approved_targets.pop(kind,None)
        elif payload.get('choice') == 'confirm':
            candidate = next((c for c in pending['candidates'] if c['id']==payload.get('candidate_id')),None)
            if not candidate:
                raise ValueError('请选择页面中已找到的候选项')
            adapter.active_page=task.page
            await adapter.observe()
            matches = [(page,item) for page,_,item in adapter.refs.values() if contains_name(item,candidate['name'])
                       and proof(page,item,candidate['name'])==candidate['id']]
            if not matches:
                raise ValueError('页面候选已变化，请修改名称重新查找后再确认')
            selected = [e['selected_option'] for e in adapter.last_view.get('elements',[])
                        if SEMESTER.fullmatch(e.get('selected_option',''))]
            if pending['term'] and selected and pending['term'] not in selected:
                raise ValueError('页面当前学期已变化，请重新核查')
            target = {**task.state['target'],kind+'_name':candidate['name']}
            if kind=='course' and pending['term']:
                target['term'] = pending['term']
            history = task.state.get('target_resolutions',[])+[{'kind':kind,'requested_name':pending['requested_name'],
                'name':candidate['name'],'term':pending['term'],'semester_fallback':pending.get('semester_fallback',False),'time':time.time()}]
            task.update(target=target,target_resolution=None,target_resolutions=history[-20:])
            adapter.approved_targets[kind] = candidate
            task.event(f'教师已确认{kind == "course" and "课程" or "作业"}“{candidate["name"]}”，继续原导航流程')
        else:
            raise ValueError('请选择确认候选或修改名称')
        adapter.target = task.state['target']
        task.observation = None
        await task._release_pointer_locked()
        task.update(status='RUNNING',control_mode='AUTO',waiting_reason='',control_epoch=task.state['control_epoch']+1,
                    auto_actions=0,navigation_failures=0,decision_failures=0)
        task.navigation_marker,task.stagnant_actions = None,0
        task.gate.set()
        return task.state
