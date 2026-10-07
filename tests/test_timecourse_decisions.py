"""Real tiny policy updates, permission boundary and coordinate bridge regression."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from lung.timecourse.data import TimeData
from lung.timecourse.protocol import VEHICLE
from lung.timecourse.decision_world import TimeDecisionWorld
from lung.timecourse.decisions import _train, _compare, run_comparison
from lung.timecourse.state_proxy import convert_coordinates
from lung.timecourse.decision_audit import audit_population
from lung.engine.config import file_hash


def artifact_snapshot(folder):
    return {str(p.relative_to(folder)):(file_hash(p),p.stat().st_mtime_ns)
            for p in folder.rglob("*") if p.is_file()}


def fixture_data():
    cells, counts = [], []
    rng = np.random.default_rng(15)
    for time in (24,72):
        for drug in (VEHICLE,"D1","D2","C1","E1","E2"):
            for _ in range(4):
                cells.append(dict(catalog_number=drug,time_point=time,dose_uM=0. if drug==VEHICLE else 1.,
                    replicate="rep1",culture_plate="plate49",author_hash_qc=True))
                counts.append(rng.integers(1,30,8))
    fp = {d:np.eye(1,2048,i,dtype=np.float32)[0] for i,d in enumerate((VEHICLE,"D1","D2","C1","E1","E2"))}
    return TimeData.from_arrays(np.array(counts),pd.DataFrame(cells),[f"g{i}" for i in range(8)],fp,
        fit_drugs=(),allowed_target_drugs=(),synthetic=True,drug_groups={"D":["D1","D2"],"C":["C1"],"E":["E1","E2"]})


class Predictor:
    def __init__(self,data):
        self.data,self.calls=data,[]
    def predict(self,cid,sample_count,seed,reference=None):
        self.calls.append(cid)
        x=reference if reference is not None else self.data.sample_controls(cid,sample_count,np.random.default_rng(seed))["control"]
        # Explicit synthetic response, actual independent RL learners train on it.
        shift=float(self.data.action_features(cid)["fingerprint"].argmax())/10
        return {"projected":x+shift}


class Proxy:
    identity="synthetic-proxy"
    def __call__(self,x):
        return float(1/(1+np.exp(-np.asarray(x).mean())))


def test_coordinate_bridge():
    actual=convert_coordinates([[2.,-1.]],[5.,4.],[3.,2.],[1.,2.],[2.,4.])
    np.testing.assert_allclose(actual,[[5.,0.]])
    with pytest.raises(ValueError):
        convert_coordinates([[0.]],[0.],[1.],[0.],[0.])


def test_training_comparison_and_role_isolation(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    data=fixture_data()
    predictor=Predictor(data)
    cfg={"device":"cpu","synthetic":True,"population":4,"projection_width":4,"epochs":2,
         "scoring_repeats":2,"audit_repeats":2,"evaluation_candidates":12,
         "daac":{"episodes_per_epoch":6,"warmup_transitions":1,"batch_size":2,"actor_delay":1,"replay_capacity":32},
         "dlpa":{"roots_per_epoch":2,"iterations":2,"candidates":8,"elites":2,"distill_updates":2,"batch_size":2}}
    ids=data.condition_ids(drugs=data.drug_groups["D"])
    world=TimeDecisionWorld(predictor,Proxy(),ids,("rep1","plate49"),"T","env",cfg)
    cfg["population_audit"]={"repeats":2,"tolerance":.5}
    assert not world.adapter.population_resolution_passed
    audit_paths=audit_population(world,cfg,tmp_path/"population_audit")
    actors, recorded={},{}
    for stage in ("daac","dlpa"):
        actor,path,counters=_train(world,cfg,stage,3,tmp_path/stage)
        assert counters["updates"]>0 and counters["parameter_delta_this_invocation"]>0
        assert Path(path).stat().st_size>0
        assert list((tmp_path/stage/"events").glob("events*"))
        assert (tmp_path/stage/"metrics.csv").stat().st_size>0
        before=artifact_snapshot(tmp_path/stage)
        resumed,_,again=_train(world,cfg,stage,3,tmp_path/stage,resume=True)
        assert again["epoch"]==2
        assert again["updates"]==counters["updates"]
        assert again["parameter_delta_this_invocation"]==0.0
        assert artifact_snapshot(tmp_path/stage)==before
        for k,v in actor.state_dict().items():
            torch.testing.assert_close(v,resumed.state_dict()[k])
        actors[stage],recorded[stage]=actor,counters
    assert set(predictor.calls)<=set(ids)
    for role in ("C","E"):
        with pytest.raises(PermissionError):
            data.sample_target(data.condition_ids(drugs=data.drug_groups[role])[0])
    eids=data.condition_ids(drugs=data.drug_groups["E"])
    evaluation=TimeDecisionWorld(predictor,Proxy(),eids,("rep1","plate49"),"E","env",cfg)
    evaluation.enable_population_audit(*audit_paths)
    report=_compare(evaluation,actors,cfg,7)
    assert set(report["enumeration"])==set(eids)|{"STOP"}
    assert {r["algorithm"] for r in report["rows"]}=={"enumeration","daac","dlpa"}
    assert all(not r["reselected_after_audit"] for r in report["rows"])
    assert not report["sampling_certificate"]
    monkeypatch.setattr("lung.timecourse.decisions.implementation_identity",lambda:"changed-code")
    with pytest.raises(ValueError,match="IDENTITY"):
        _train(world,cfg,"daac",3,tmp_path/"daac",resume=True)
    (tmp_path/"observed.json").write_text(json.dumps({"updates":recorded,"comparison":report},indent=2))


def test_comparison_entry_freezes_before_e_labels(tmp_path,monkeypatch):
    data=fixture_data()
    predictor=Predictor(data)
    monkeypatch.setattr("lung.timecourse.nested.FrozenTimePredictor",lambda *args:predictor)
    monkeypatch.setattr("lung.timecourse.decisions.FrozenStateProxy",lambda *args:Proxy())
    reads=[]
    original=TimeData.sample_target
    def checked_read(self,cid,*args,**kwargs):
        row=self.conditions.loc[cid]
        assert row.catalog_number in self.drug_groups["E"]
        for stage in ("daac","dlpa"):
            saved=torch.load(tmp_path/"run"/"rep1_plate49"/"seed_0"/stage/"last.pt",weights_only=False)
            assert saved["epoch"]==1
        reads.append(cid)
        return original(self,cid,*args,**kwargs)
    monkeypatch.setattr(TimeData,"sample_target",checked_read)
    cfg={"synthetic":True,"device":"cpu","epochs":1,"policy_seeds":[0],"population":3,
         "population_audit":{"repeats":2,"tolerance":.5},
         "projection_width":4,"scoring_repeats":1,"audit_repeats":2,"evaluation_candidates":8,
         "log_root":str(tmp_path/"logs"),"log_anchor":str(tmp_path/"run"),
         "daac":{"episodes_per_epoch":4,"warmup_transitions":1,"batch_size":2,"actor_delay":1,"replay_capacity":16},
         "dlpa":{"roots_per_epoch":1,"iterations":1,"candidates":6,"elites":2,"distill_updates":1,"batch_size":2}}
    def interrupted_compare(*args):
        raise RuntimeError("simulated_post_training_interruption")
    monkeypatch.setattr("lung.timecourse.decisions._compare",interrupted_compare)
    with pytest.raises(RuntimeError,match="simulated_post_training_interruption"):
        run_comparison(tmp_path,cfg,{"artifacts":[]},{},tmp_path/"run")
    assert reads==[]
    preserved={stage:artifact_snapshot(tmp_path/"run"/"rep1_plate49"/"seed_0"/stage)
               for stage in ("daac","dlpa")}
    events_before=artifact_snapshot(tmp_path/"logs")
    monkeypatch.setattr("lung.timecourse.decisions._compare",_compare)
    result=run_comparison(tmp_path,cfg,{"artifacts":[]},{},tmp_path/"run",resume=True)
    for stage in ("daac","dlpa"):
        assert artifact_snapshot(tmp_path/"run"/"rep1_plate49"/"seed_0"/stage)==preserved[stage]
    assert artifact_snapshot(tmp_path/"logs")==events_before
    assert result["status"]=="complete" and len(result["weights"])==2
    assert len(reads)==2
    assert all(Path(p).is_absolute() and Path(p).stat().st_size>0 for p in result["artifacts"])
    report=json.loads(Path(result["report"]).read_text())
    assert report["E_target_access"]=="after_all_training_frozen"
    assert report["calibration_reward_access"] is False
    assert list((tmp_path/"logs").rglob("events.out*"))
    assert all(row["measured_proxy_rank"]>=1 for row in report["results"][0]["comparison"]["rows"])


def test_population_gate_fails_closed(tmp_path,monkeypatch):
    data=fixture_data()
    ids=data.condition_ids(drugs=data.drug_groups["D"])
    cfg={"population":2,"projection_width":4,"synthetic":True,
         "population_audit":{"repeats":2,"tolerance":.01}}
    world=TimeDecisionWorld(Predictor(data),Proxy(),ids,("rep1","plate49"),"T","env",cfg)
    with pytest.raises(Exception,match="POPULATION_RESOLUTION_UNRESOLVED"):
        world.adapter.validate_ready()
    monkeypatch.setattr(world,"raw_score",lambda root,plan,seeds,population:torch.full((len(seeds),),population/10.))
    with pytest.raises(ValueError,match="POPULATION_AUDIT"):
        audit_population(world,cfg,tmp_path/"audit")
    assert not world.adapter.population_resolution_passed
    report=json.loads((tmp_path/"audit"/"report.json").read_text())
    assert report["passed"] is False
    report["passed"]=True
    (tmp_path/"audit"/"report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError,match="POPULATION_AUDIT_NOT_PASSED"):
        world.enable_population_audit(tmp_path/"audit"/"registration.json",tmp_path/"audit"/"report.json")
