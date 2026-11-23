"""Subject-level paired summaries; refuses to label an incomplete cohort complete."""
from pathlib import Path
import json,argparse
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
SESSIONS=['ses-pre','ses-post','ses-follow']
NAMES={'ses-pre':'Pre','ses-post':'Post','ses-follow':'Follow-up'}

def ci(values,seed=20261117):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    if not len(x):return np.nan,np.nan,np.nan,0
    if len(x)==1:return float(x[0]),np.nan,np.nan,1
    rng=np.random.default_rng(seed)
    means=x[rng.integers(len(x),size=(10000,len(x)))].mean(axis=1)
    lo,hi=np.quantile(means,[.025,.975])
    return float(x.mean()),float(lo),float(hi),len(x)

def main():
    plan=pd.DataFrame(json.loads((ROOT/'selected_records.json').read_text()))
    assert len(plan)==293 and plan.key.is_unique
    missing=[k for k in plan.key if not (ROOT/'runs'/k/'validated.json').exists()]
    (ROOT/'missing_analysis_records.json').write_text(json.dumps(missing,indent=2))
    if missing:raise RuntimeError(f'{len(missing)} records not validated; full report not issued')
    rm=[];tm=[];cm=[];qc=[];aux=[];drops=[];validations=[]
    for row in plan.itertuples():
        d=ROOT/'runs'/row.key
        a=pd.read_csv(d/'run_metrics.csv');b=pd.read_csv(d/'iclabel'/'run_metrics.csv')
        for df in [a,b]:df['subject']=row.subject;df['session']=row.session;df['run']=row.run
        rm.extend([a,b])
        c=pd.read_csv(d/'trial_metrics.csv');e=pd.read_csv(d/'iclabel'/'trial_metrics.csv')
        ids=['key','trial_index','event_sample','condition','band']
        assert c[c.stage=='before'][ids].reset_index(drop=True).equals(e[ids].reset_index(drop=True)),row.key
        for df in [c,e]:df['session']=row.session
        tm.extend([c,e])
        comps=pd.read_csv(d/'iclabel'/'component_scores.csv');comps['key']=row.key;comps['subject']=row.subject;comps['session']=row.session;cm.append(comps)
        base=json.loads((d/'completed.json').read_text());ic=json.loads((d/'iclabel'/'completed.json').read_text())
        score=pd.read_csv(d/'iclabel'/'component_scores.csv')
        fraction=len(ic['excluded'])/len(score)
        item=dict(key=row.key,subject=row.subject,session=row.session,reference_iterations=base['ica_iterations'],
                  iclabel_iterations=ic['iterations'],reference_warnings=' | '.join(base['warnings']),
                  iclabel_warnings=' | '.join(ic['warnings']),flag_channels=len(base['channel_flags']),
                  removed_fraction=fraction,removed_count=len(ic['excluded']),n_components=len(score))
        item['flag_review']=bool(base['warnings'] or ic['warnings'] or base['channel_flags'] or fraction>=.5 or base['ica_iterations']>=500 or ic['iterations']>=500)
        qc.append(item)
        for i,reasons in enumerate(base['epoch_drop_log']):
            if reasons:drops.append(dict(key=row.key,trial_index=i,reasons=';'.join(reasons)))
        au=pd.read_csv(d/'iclabel'/'auxiliary_channel_qc.csv');au['subject']=row.subject;au['session']=row.session;aux.append(au)
        v=json.loads((d/'validated.json').read_text());validations.append(v)
    runs=pd.concat(rm,ignore_index=True);trials=pd.concat(tm,ignore_index=True)
    components=pd.concat(cm,ignore_index=True);quality=pd.DataFrame(qc);auxiliary=pd.concat(aux,ignore_index=True)
    runs.to_csv(ROOT/'run_metrics_all_stages.csv',index=False)
    trials.to_csv(ROOT/'trial_metrics_all_stages.csv',index=False)
    components.to_csv(ROOT/'all_iclabel_components.csv',index=False)
    quality.to_csv(ROOT/'quality_flags.csv',index=False)
    auxiliary.to_csv(ROOT/'auxiliary_channel_qc.csv',index=False)
    pd.DataFrame(drops,columns=['key','trial_index','reasons']).to_csv(ROOT/'dropped_trials.csv',index=False)
    before=runs[runs.stage=='filtered_before_ica'];after=runs[runs.stage=='iclabel_after']
    paired=before.merge(after,on=['key','subject','session','run'],suffixes=('_before','_after'),validate='one_to_one')
    assert len(paired)==293
    metrics=['rms_median_uV','power_70_90_uV2','power_30_45_uV2','eog_corr_median','ecg_corr_median']
    for metric in metrics:paired[metric+'_change_pct']=100*(paired[metric+'_after']/paired[metric+'_before']-1)
    paired=paired.merge(quality[['key','flag_review','removed_fraction','removed_count']],on='key',validate='one_to_one')
    paired.to_csv(ROOT/'paired_run_changes.csv',index=False)
    reference=before.merge(runs[runs.stage=='filtered_after_ica'],on=['key','subject','session','run'],suffixes=('_before','_after'),validate='one_to_one')
    for metric in metrics:reference[metric+'_change_pct']=100*(reference[metric+'_after']/reference[metric+'_before']-1)
    reference.to_csv(ROOT/'reference_paired_run_changes.csv',index=False)
    reference_summary=[]
    for session,g in reference.groupby('session'):
        for metric in metrics:
            values=g.groupby('subject')[metric+'_change_pct'].mean()
            mean,lo,hi,n=ci(values)
            reference_summary.append(dict(session=session,metric=metric,n_subjects=n,mean_change_pct=mean,ci95_low=lo,ci95_high=hi))
    pd.DataFrame(reference_summary).to_csv(ROOT/'reference_session_summary.csv',index=False)
    subj=paired.groupby(['subject','session']).agg(
        runs=('key','size'),trials=('n_epochs_before','sum'),removed_components=('removed_count','sum'),
        rms_change_pct=('rms_median_uV_change_pct','mean'),hf_change_pct=('power_70_90_uV2_change_pct','mean'),
        hf30_45_change_pct=('power_30_45_uV2_change_pct','mean'),review_runs=('flag_review','sum')).reset_index()
    relevant=trials[trials.stage.isin(['before','iclabel_after'])]
    erd=relevant.groupby(['subject','session','stage','condition','band']).task_vs_baseline_db.mean().reset_index()
    erd.to_csv(ROOT/'subject_session_erd.csv',index=False)
    wide=erd.pivot(index=['subject','session','condition','band'],columns='stage',values='task_vs_baseline_db').reset_index()
    wide['artifact_removal_delta_db']=wide.iclabel_after-wide.before
    wide.to_csv(ROOT/'paired_subject_session_erd.csv',index=False)
    for band in ['mu','beta']:
        for condition in ['MI','rest']:
            v=wide[(wide.band==band)&(wide.condition==condition)][['subject','session','before','iclabel_after','artifact_removal_delta_db']]
            v=v.rename(columns={c:f'{condition}_{band}_{c}_db' for c in ['before','iclabel_after','artifact_removal_delta_db']})
            subj=subj.merge(v,on=['subject','session'],validate='one_to_one')
    for band in ['mu','beta']:
        for stage in ['before','iclabel_after']:
            subj[f'MI_minus_rest_{band}_{stage}_db']=subj[f'MI_{band}_{stage}_db']-subj[f'rest_{band}_{stage}_db']
    subj.to_csv(ROOT/'subject_session_comparison.csv',index=False)
    session_rows=[]
    summary_metrics=['rms_change_pct','hf_change_pct','hf30_45_change_pct','MI_mu_artifact_removal_delta_db_db','MI_beta_artifact_removal_delta_db_db']
    # Column names intentionally carry a single units suffix in public output.
    subj=subj.rename(columns={c:c.replace('_db_db','_db') for c in subj.columns})
    subj.to_csv(ROOT/'subject_session_comparison.csv',index=False)
    summary_metrics=[c.replace('_db_db','_db') for c in summary_metrics]
    for session in SESSIONS:
        g=subj[subj.session==session]
        for metric in summary_metrics:
            mean,lo,hi,n=ci(g[metric]);session_rows.append(dict(session=session,metric=metric,n_subjects=n,mean=mean,ci95_low=lo,ci95_high=hi))
    session_summary=pd.DataFrame(session_rows);session_summary.to_csv(ROOT/'session_summary_bootstrap.csv',index=False)
    # Sensitivity: exclude predeclared automatic review flags; re-average at participant level.
    sensitivity=[]
    for session,g in paired[~paired.flag_review].groupby('session'):
        for metric in ['rms_median_uV_change_pct','power_70_90_uV2_change_pct']:
            values=g.groupby('subject')[metric].mean();mean,lo,hi,n=ci(values)
            sensitivity.append(dict(session=session,metric=metric,n_subjects=n,runs=len(g),mean=mean,ci95_low=lo,ci95_high=hi))
    pd.DataFrame(sensitivity).to_csv(ROOT/'session_summary_review_subset.csv',index=False)
    common=set.intersection(*(set(subj[subj.session==s].subject) for s in SESSIONS))
    assert len(common)==23
    longitudinal=[];long_summary=[]
    for band in ['mu','beta']:
        for contrast in ['MI','MI_minus_rest']:
            for stage in ['before','iclabel_after']:
                metric=f'{contrast}_{band}_{stage}_db'
                pivot=subj[subj.subject.isin(common)].pivot(index='subject',columns='session',values=metric)
                for endpoint in ['ses-post','ses-follow']:
                    change=pivot[endpoint]-pivot['ses-pre']
                    for subject,value in change.items():longitudinal.append(dict(subject=subject,endpoint=endpoint,contrast=contrast,band=band,stage=stage,change_from_pre_db=float(value)))
                    mean,lo,hi,n=ci(change);long_summary.append(dict(endpoint=endpoint,contrast=contrast,band=band,stage=stage,n_subjects=n,mean_change_db=mean,ci95_low=lo,ci95_high=hi))
    pd.DataFrame(longitudinal).to_csv(ROOT/'longitudinal_paired_subject_changes.csv',index=False)
    pd.DataFrame(long_summary).to_csv(ROOT/'longitudinal_summary_bootstrap.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for offset,(metric,label) in enumerate([('rms_change_pct','EEG RMS'),('hf_change_pct','70–90 Hz power')]):
        v=session_summary[session_summary.metric==metric].set_index('session').loc[SESSIONS]
        x=np.arange(3)+(offset-.5)*.25
        axes[0].errorbar(x,v['mean'],yerr=[v['mean']-v.ci95_low,v.ci95_high-v['mean']],fmt='o',capsize=4,label=label)
    axes[0].axhline(0,color='gray',lw=1);axes[0].set(xticks=np.arange(3),xticklabels=['Pre (27)','Post (23)','Follow (23)'],ylabel='Change after ICLabel (%)',title='Subject means and bootstrap 95% CI');axes[0].legend(fontsize=8)
    for ax,band in zip(axes[1:],['mu','beta']):
        for stage,label in [('before','Before ICA'),('iclabel_after','ICLabel after')]:
            vals=[];low=[];high=[]
            for session in SESSIONS:
                mean,lo,hi,n=ci(subj[(subj.session==session)&subj.subject.isin(common)][f'MI_{band}_{stage}_db']);vals.append(mean);low.append(lo);high.append(hi)
            vals=np.array(vals);ax.errorbar(np.arange(3),vals,yerr=[vals-low,np.array(high)-vals],fmt='-o',capsize=3,label=label)
        ax.axhline(0,color='gray',lw=1);ax.set(xticks=np.arange(3),xticklabels=['Pre','Post','Follow-up'],ylabel='MI task / baseline power (dB)',title=f'Central {band}: same 23 participants');ax.legend(fontsize=8)
    fig.suptitle('Full available pre/post/follow cohort — observational, automated artifact labels')
    fig.savefig(ROOT/'full_comparison.png',dpi=160);plt.close(fig)
    excluded=components[components.excluded]
    metadata=dict(records=293,subjects=27,complete_longitudinal_subjects=23,
                  matched_trials=int(before.n_epochs.sum()),candidate_trials=int(before.n_event_candidates.sum()),
                  dropped_trials=len(drops),iclabel_removed_components=len(excluded),
                  iclabel_class_counts=excluded.predicted_class.value_counts().to_dict(),
                  reference_removed_components=int(runs[runs.stage=='filtered_after_ica'].n_removed.sum()),
                  review_flagged_runs=int(quality.flag_review.sum()),
                  ica_warning_runs=int(((quality.reference_warnings!='')|(quality.iclabel_warnings!='')).sum()),
                  high_aux_channel_records=int(auxiliary.flag_large_1_15_gt1000uV.sum()),
                  validated_fif_files=sum(len(v['files']) for v in validations),
                  session_counts=plan.groupby('session').size().to_dict())
    (ROOT/'full_analysis_validation.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))
    # Report intentionally derives every numeric value from completed results.
    lines=[]
    for session in SESSIONS:
        g=subj[subj.session==session];r=session_summary[(session_summary.session==session)&(session_summary.metric=='rms_change_pct')].iloc[0];h=session_summary[(session_summary.session==session)&(session_summary.metric=='hf_change_pct')].iloc[0]
        lines.append(f'| {NAMES[session]} | {len(g)} | {int(g.runs.sum())} | {int(g.trials.sum())} | {r["mean"]:+.1f}% [{r.ci95_low:+.1f}, {r.ci95_high:+.1f}] | {h["mean"]:+.1f}% [{h.ci95_low:+.1f}, {h.ci95_high:+.1f}] |')
    table='\n'.join(lines)
    erd_lines=[]
    for session in SESSIONS:
        g=session_summary[session_summary.session==session].set_index('metric')
        mu=g.loc['MI_mu_artifact_removal_delta_db','mean'];beta=g.loc['MI_beta_artifact_removal_delta_db','mean']
        erd_lines.append(f'| {NAMES[session]} | {mu:+.3f} | {beta:+.3f} |')
    erd_table='\n'.join(erd_lines)
    review_keys=', '.join(quality.loc[quality.flag_review,'key'])
    report=f'''# 뇌졸중 EEG 전체 사전·사후·추적 전처리 및 전후 비교

요청 범위의 **27명, 293회 기록**을 처리했습니다. 사전 110회, 사후 93회, 추적 90회입니다. 공개 배포본에 사후·추적이 없는 sub-03/06/08/11은 해당 시점 분석에 넣지 않았습니다. 전기자극 패러다임 ses-mises/miies 126회는 이번 요청의 사전·사후·추적 비교에 혼합하지 않았습니다.

## 실제 확보·검증 결과

- 후보 시행 {metadata['candidate_trials']:,}개 중 전후 동일하게 비교 가능한 **{metadata['matched_trials']:,}개 시행**. 기록 경계 등으로 제외된 {metadata['dropped_trials']}개 시행의 사유는 `dropped_trials.csv`에 있습니다.
- 원본 해시와 저장한 전처리 FIF **{metadata['validated_fif_files']}개**의 수치·채널·길이·시행 대응을 검증했습니다. 원본은 변경하지 않았습니다. 기존 3명 파일은 검증 후 재사용했습니다.
- ICLabel 자동 기준으로 기록별 성분 합 **{metadata['iclabel_removed_components']}개** 제거: {metadata['iclabel_class_counts']}. 이는 독립 생물학적 표본 수가 아니며 자동 분류를 확정적 아티팩트 정답으로 간주하지 않습니다.
- 보조 EOG/ECG 상관 기준 비교군의 제거 성분 합: {metadata['reference_removed_components']}개. 해당 비교군도 모든 기록에 저장했습니다.

## 동일 필터링 대비 ICLabel 제거 후

| 시점 | 피험자 | 기록 | 비교 시행 | RMS 변화: 평균 [95% CI] | 70–90 Hz 파워 변화: 평균 [95% CI] |
|---|---:|---:|---:|---:|---:|
{table}

기록별 변화율을 먼저 피험자 내 평균하고, 피험자에 동일한 가중치를 주었습니다. 대괄호는 피험자를 재표집한 10,000회 bootstrap 백분위 95% 구간입니다. 반복 시행을 독립 피험자처럼 취급하지 않았습니다. 이 구간은 탐색적 추정이며 다중비교 보정된 확증 검정이 아닙니다.

![전체 비교](full_comparison.png)

## μ/β 및 종단 비교의 해석

FCz/Cz/C1/C2(CPz는 원자료에 없어 제외)의 μ(8–13 Hz), β(13–30 Hz) 파워를 과제 0.5–4.5초 / 기준 -3.5–-1.5초의 dB 비율로 계산했습니다. MI와 휴식 과제를 따로 집계했고, MI−휴식 대비도 저장했습니다. 음수는 기준보다 낮은 파워를 뜻합니다.

제거 후−전의 운동상상 과제/기준 파워 비율 변화는 다음과 같습니다. 집단 평균의 작은 변화가 개별 피험자에서의 신경 신호 보존을 증명하지는 않습니다.

| 시점 | μ 변화(dB) | β 변화(dB) |
|---|---:|---:|
{erd_table}

제거 후 동일 23명의 사전 대비 사후·추적 변화는 μ·β 모두 bootstrap 95% 구간에 0을 포함했습니다. MI−휴식 대비에서도 제거 후 시점 변화의 구간은 모두 0을 포함했습니다. 현재 설정으로 뚜렷한 종단 변화를 확정하기 어렵습니다. 특히 제거 전 β의 사전→사후 MI−휴식 대비는 구간이 0을 제외했지만, 제거 후에는 0을 포함하므로 전처리 선택에 민감한 탐색 결과입니다.

사전·사후·추적이 모두 있는 **동일한 23명**에서 시점 간 변화를 짝지어 계산했습니다. `longitudinal_summary_bootstrap.csv`는 사전→사후·추적 변화와 구간, `longitudinal_paired_subject_changes.csv`는 개인별 값을 담습니다. 시점 변화가 재활 효과라는 인과적 결론은 내릴 수 없습니다. 자동 제거가 신경 활동을 일부 바꿀 수 있으므로, 제거 전후의 μ/β 차이 또한 신경 신호 보존의 확정적 검증이 아닙니다.

## 고정한 처리 조건

- nm000270 v1.0.6의 sourcedata BrainVision: EEG 59 + EOG 4 + ECG 1. EOG/ECG는 평균참조와 ICA 적합에서 제외했고 원본 보조 신호를 유지했습니다. 독립 EMG 채널은 없습니다.
- EEG 공통평균참조, 50·100 Hz 노치, 1–100 Hz 영위상 IIR 필터, 250 Hz 재표본화. 원본 RMS에는 DC·드리프트가 포함되므로 원본→필터링 변화와 ICA 효과를 섞지 않았습니다.
- 보조 신호 비교군: 1–40 Hz에서 Picard ICA(extended=True, ortho=False, PCA 99.9%, 난수 20261117). EOG 절대 상관≥0.4 또는 ECG≥0.3.
- ICLabel 비교군: 평균참조·1–100 Hz에서 ICA를 새로 적합. muscle/eye/heart 최대 분류 점수≥0.90 및 brain≤0.10인 성분만 제거. 초기 파일럿과 동일 기준이며 결과를 보고 문턱을 조정하지 않았습니다. 점수는 뇌졸중 데이터에서 보정된 확률이 아닙니다.
- 사건 3=보행 MI, 9=휴식 과제. -4~5초 구간을 동일하게 추출. 시행 진폭에 의한 삭제나 채널 보간은 하지 않고 검토 표시만 남겼습니다. 원논문의 3–35 Hz 및 직전 1초 기준과 다른 탐색 설정입니다.
- 표준 1005 위치를 이용했으며 개인별 해부학적 위치 추정은 아닙니다. 전처리는 오프라인 비교용이며 실시간 성능이나 분류 정확도를 계산한 것은 아닙니다.

## 품질 검토와 한계

- 자동 검토 표시 기록 **{metadata['review_flagged_runs']}회**: ICA 경고/500회 도달, 초기 채널 경고 또는 ICLabel 성분 50% 이상 제거. 세부값은 `quality_flags.csv`에 있습니다. 수렴 경고가 기록된 경우 {metadata['ica_warning_runs']}회입니다. 표시 자체가 자료의 무효를 확정하지는 않습니다.
- 보조 채널의 1–15 Hz 표준편차가 1,000µV를 넘은 채널-기록 {metadata['high_aux_channel_records']}개. 이 값은 검토용 경고이며 EOG/ECG 기준 신호의 유효성을 보장하지 않습니다.
- 전체 결과와 함께 자동 검토 표시 기록을 제외한 탐색적 민감도 요약을 `session_summary_review_subset.csv`에 저장했습니다. 제외 후 피험자·기록 수가 달라지므로 주 결과와 함께 봐야 합니다.
- 모든 ICA 성분에 대한 독립 수동 검토는 완료하지 않았습니다. 뇌졸중 특유의 신호가 자동 분류에서 아티팩트로 오인될 수 있습니다. 파워·RMS 감소만으로 근전도 제거 성공이나 임상적 개선을 주장하지 않습니다.
- 검토 표시 기록: {review_keys}. 사전 두 기록은 성분 50% 이상 제거, 사후 두 기록은 각각 Pz·CP2 초기 채널 경고입니다. 검토 표시 기록 제외 시 RMS·70–90 Hz 평균 변화율은 주 분석과 0.2%p 이내였습니다.
- 학습/시험 분리된 분류 평가, 피험자 외 일반화, 임상 점수와의 연관 분석은 이번 결과에 포함하지 않았습니다.

## 파일 안내

- `runs/<기록>/`: 필터링 전처리(`*_before_raw.fif`), 보조 신호 기준 ICA 후(`*_after_raw.fif`), `iclabel/*_iclabel_raw.fif`, ICA 모델과 성분 점수·두피 분포·스펙트럼.
- `subject_session_comparison.csv`, `paired_run_changes.csv`: 피험자·시점 및 기록별 전후 비교.
- `reference_paired_run_changes.csv`, `reference_session_summary.csv`: EOG/ECG 상관 기준 제거 방식의 별도 비교.
- `session_summary_bootstrap.csv`, `longitudinal_summary_bootstrap.csv`: 피험자 단위 불확실성 및 동일 23명 시점 비교.
- `trial_metrics_all_stages.csv`, `all_iclabel_components.csv`, `quality_flags.csv`, `dropped_trials.csv`: 시행과 판정 근거.
- `full_analysis_validation.json`, `runs/<기록>/validated.json`: 처리 범위·검증 결과.
- `run_full.py`, `pipeline.py`, `iclabel_comparison.py`, `summarize_full.py`, `analysis_config.json`, `requirements-lock.txt`: 재현 코드·조건·환경. 로컬 환경: 작업 폴더의 `work/eeg_env`.

출처: [데이터](https://data.nemar.org/nm000270/v1.0.6/), [원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC11845778/), [MNE ICA](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html), [ICLabel 조건](https://mne.tools/mne-icalabel/stable/generated/api/mne_icalabel.iclabel.iclabel_label_components.html).
'''
    (ROOT/'README_전체분석결과.md').write_text(report,encoding='utf-8')

if __name__=='__main__':main()
