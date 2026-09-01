#!/usr/bin/env python3
"""Project replicated and cross-etiology mouse withdrawal data into resolver-lock space."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT=Path(__file__).resolve().parents[2]
COUNTS=REPO_ROOT/"data/mouse_pseudobulk_derived/pseudobulk_counts.tsv.gz"
META=REPO_ROOT/"data/mouse_pseudobulk_derived/pseudobulk_metadata.tsv"
POOLED=REPO_ROOT/"data/mouse_pseudobulk_derived/state_module_trajectories.tsv"
OUTPUT=REPO_ROOT/"results/S13_mouse_resolver_retention_trajectory"
COLORS={"control":"#2A9D8F","fibrosis":"#C43D55","regression":"#E5A93D"}

MODULES={
    "HSC_retention":("HSC_Mesenchymal",["Itgb5","Gas7","Spon1","Svep1","Pdgfra","Lama2","Epha3","Dpysl3","Ltbp2","Efemp1","Serpine1"]),
    "feedback_core":("HSC_Mesenchymal",["Serpine1","Spon1","Gas7","Ltbp2"]),
    "TGF_latch":("HSC_Mesenchymal",["Tgfb1","Tgfbr1","Tgfbr2","Smad3","Smad7","Ctgf","Ccn2","Thbs1"]),
    "antiproteolytic_brake":("HSC_Mesenchymal",["Serpine1","Timp1","Timp2","Thbs1"]),
    "protease_potential":("Myeloid",["Plau","Plat","Mmp2","Mmp9","Mmp14"]),
    "Endo_resolver":("Endothelial",["Wnt9b","Wnt7b","Vwf","Ace","Kdr","Esam","Emcn","Plvap","Klf2","Nr2f2"]),
    "LAM_resolver":("Myeloid",["Trem2","Gpnmb","Lpl","Mmp14","Ctsb","Ctsd","Axl","Mertk"]),
}


def zscore(values: np.ndarray) -> np.ndarray:
    return (values-values.mean())/max(values.std(ddof=1),1e-8)


def build_scores() -> tuple[pd.DataFrame,pd.DataFrame]:
    counts=pd.read_csv(COUNTS,sep="\t",index_col=0)
    meta=pd.read_csv(META,sep="\t")
    audit=[]; score={}
    for module,(cell_type,genes) in MODULES.items():
        local=meta.loc[meta.cell_type.eq(cell_type)].sort_values("sample")
        columns=local.pseudobulk.tolist()
        matrix=counts[columns]
        library=matrix.sum(axis=0).to_numpy(float)
        logcpm=np.log2((matrix.to_numpy(float)+.5)/(library[None,:]+1)*1e6)
        expression=pd.DataFrame(logcpm,index=matrix.index,columns=local["sample"])
        present=list(dict.fromkeys(gene for gene in genes if gene in expression.index))
        gene_z=np.vstack([zscore(expression.loc[gene].to_numpy(float)) for gene in present])
        score[module]=gene_z.mean(axis=0)
        audit.append({"module":module,"cell_type":cell_type,"requested":len(set(genes)),"present":len(present),"genes":";".join(present)})
    samples=meta[["sample","condition"]].drop_duplicates().sort_values("sample").reset_index(drop=True)
    for module,values in score.items(): samples[module]=values
    samples["clearance_inefficiency"]=zscore(samples.antiproteolytic_brake.to_numpy())-zscore(samples.protease_potential.to_numpy())
    lock_features=["HSC_retention","feedback_core","TGF_latch","clearance_inefficiency"]
    resolver_features=["Endo_resolver","LAM_resolver"]
    samples["lock_axis"]=np.mean([zscore(samples[name].to_numpy()) for name in lock_features],axis=0)
    samples["resolver_axis"]=np.mean([zscore(samples[name].to_numpy()) for name in resolver_features],axis=0)
    samples["control_imbalance"]=(samples.lock_axis-samples.resolver_axis)/np.sqrt(2)
    return samples,pd.DataFrame(audit)


def exact_group_test(samples:pd.DataFrame,outcome:str) -> pd.DataFrame:
    values=samples[outcome].to_numpy(float)
    observed={}
    means=samples.groupby("condition")[outcome].mean()
    observed["progression"]=means.fibrosis-means.control
    observed["withdrawal"]=means.regression-means.fibrosis
    observed["residual"]=means.regression-means.control
    null={name:[] for name in observed}
    indices=set(range(9))
    for control in itertools.combinations(range(9),3):
        remaining=indices-set(control)
        for fibrosis in itertools.combinations(sorted(remaining),3):
            regression=sorted(remaining-set(fibrosis))
            m0=values[list(control)].mean(); m1=values[list(fibrosis)].mean(); m2=values[regression].mean()
            null["progression"].append(m1-m0); null["withdrawal"].append(m2-m1); null["residual"].append(m2-m0)
    rows=[]
    for contrast,effect in observed.items():
        distribution=np.asarray(null[contrast])
        rows.append({"outcome":outcome,"contrast":contrast,"effect":float(effect),"exact_two_sided_p":float((1+np.sum(np.abs(distribution)>=abs(effect)))/(len(distribution)+1)),"exact_directional_p":float((1+np.sum(distribution>=effect))/(len(distribution)+1)) if effect>=0 else float((1+np.sum(distribution<=effect))/(len(distribution)+1)),"n_assignments":len(distribution)})
    return pd.DataFrame(rows)


def geometry(samples:pd.DataFrame) -> dict:
    centroids=samples.groupby("condition")[["resolver_axis","lock_axis"]].mean()
    control=centroids.loc["control"].to_numpy(); fibrosis=centroids.loc["fibrosis"].to_numpy(); regression=centroids.loc["regression"].to_numpy()
    progression=fibrosis-control; recovery=regression-fibrosis
    cosine=float(np.dot(-progression,recovery)/(np.linalg.norm(progression)*np.linalg.norm(recovery)))
    return {
        "progression_vector":progression.tolist(),"withdrawal_vector":recovery.tolist(),
        "return_cosine":cosine,"return_angle_degrees":float(np.degrees(np.arccos(np.clip(cosine,-1,1)))),
        "residual_distance_ratio":float(np.linalg.norm(regression-control)/np.linalg.norm(fibrosis-control)),
        "control_centroid":control.tolist(),"fibrosis_centroid":fibrosis.tolist(),"regression_centroid":regression.tolist(),
    }


def pooled_trajectories() -> pd.DataFrame:
    d=pd.read_csv(POOLED,sep="\t")
    p=d.pivot_table(index=["sample","etiology","timepoint","week"],columns="module",values="mean").reset_index()
    p["resolver_proxy"]=p[["Endo4_WNT_resolver","LAM_resolver"]].mean(axis=1)
    p["lock_proxy"]=p["HSC_pathogenic_retention"]
    for column in ["resolver_proxy","lock_proxy"]:
        p[column]=zscore(p[column].to_numpy(float))
    return p


def plot(samples:pd.DataFrame,pooled:pd.DataFrame,geom:dict) -> None:
    sns.set_theme(style="ticks",context="paper",font_scale=1.0)
    fig=plt.figure(figsize=(14.5,10.5))
    grid=fig.add_gridspec(2,2,left=.07,right=.98,bottom=.07,top=.91,wspace=.30,hspace=.34)
    ax=fig.add_subplot(grid[0,0])
    for condition,local in samples.groupby("condition",observed=True): ax.scatter(local.resolver_axis,local.lock_axis,s=58,color=COLORS[condition],label=condition,edgecolor="white",linewidth=.7)
    cent=samples.groupby("condition")[["resolver_axis","lock_axis"]].mean()
    for start,end in [("control","fibrosis"),("fibrosis","regression")]: ax.annotate("",xy=cent.loc[end],xytext=cent.loc[start],arrowprops=dict(arrowstyle="-|>",lw=2.3,color="#424A4E"))
    ax.axline((0,0),slope=1,color="#7E898E",ls="--",lw=1)
    ax.set(xlabel="Resolver capacity",ylabel="HSC lock load")
    ax.set_title("A  Replicated withdrawal follows a non-reciprocal return path",loc="left",fontweight="bold"); ax.legend(frameon=False)

    ax=fig.add_subplot(grid[0,1])
    long=samples.melt(id_vars=["sample","condition"],value_vars=["resolver_axis","lock_axis","control_imbalance"],var_name="axis",value_name="score")
    sns.boxplot(data=long,x="condition",y="score",hue="axis",palette={"resolver_axis":"#3182A6","lock_axis":"#B83D4B","control_imbalance":"#765A89"},fliersize=0,ax=ax)
    sns.stripplot(data=long,x="condition",y="score",hue="axis",dodge=True,palette={"resolver_axis":"#3182A6","lock_axis":"#B83D4B","control_imbalance":"#765A89"},size=4,ax=ax,legend=False)
    ax.axhline(0,color="#777",lw=.8); ax.set(xlabel="",ylabel="Standardized control score")
    ax.set_title("B  Lock load remains displaced after injury withdrawal",loc="left",fontweight="bold"); ax.legend(frameon=False,ncol=3,fontsize=7)

    ax=fig.add_subplot(grid[1,0]); ax.set_axis_off(); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.text(.03,.90,"CONTROL",color=COLORS['control'],fontweight="bold",fontsize=12)
    ax.annotate("",xy=(.48,.72),xytext=(.18,.82),arrowprops=dict(arrowstyle="-|>",lw=6,color=COLORS['fibrosis']))
    ax.text(.50,.70,"FIBROSIS",color=COLORS['fibrosis'],fontweight="bold",fontsize=12)
    ax.annotate("",xy=(.73,.45),xytext=(.53,.66),arrowprops=dict(arrowstyle="-|>",lw=6,color=COLORS['regression']))
    ax.text(.75,.42,"REGRESSION",color="#A87513",fontweight="bold",fontsize=12)
    ax.plot([.75,.12],[.35,.75],ls="--",color="#879198",lw=1.2)
    ax.text(.53,.20,f"Return angle  {geom['return_angle_degrees']:.1f}°\nResidual distance  {geom['residual_distance_ratio']:.2f}× peak displacement",ha="center",fontsize=11,bbox=dict(boxstyle="round,pad=.45",fc="white",ec="#919BA0"))
    ax.set_title("C  Incomplete withdrawal displacement",loc="left",fontweight="bold")

    ax=fig.add_subplot(grid[1,1])
    baseline=pooled.loc[pooled.etiology.eq("Untreated")].iloc[0]
    ax.scatter(baseline.resolver_proxy,baseline.lock_proxy,s=90,color="#626B70",marker="s",label="untreated")
    for etiology,color in [("CCl4","#3C78A8"),("FAT_MASH","#B64554")]:
        local=pooled.loc[pooled.etiology.eq(etiology)].sort_values("week")
        ax.plot(local.resolver_proxy,local.lock_proxy,marker="o",lw=2.3,color=color,label=etiology)
        for _,row in local.iterrows(): ax.text(row.resolver_proxy+.03,row.lock_proxy+.03,row.timepoint,fontsize=7,color=color)
    ax.axline((0,0),slope=1,color="#7E898E",ls="--",lw=1)
    ax.set(xlabel="Resolver proxy",ylabel="HSC-retention proxy")
    ax.set_title("D  Cross-etiology trajectories converge on late resolver loss",loc="left",fontweight="bold"); ax.legend(frameon=False)
    for axis in fig.axes:
        if axis.axison: axis.spines[["top","right"]].set_visible(False)
    fig.suptitle("Mouse withdrawal trajectories connect temporal non-reciprocity to the human resolver–lock landscape",fontsize=16,fontweight="bold")
    fig.savefig(OUTPUT/"Mouse_resolver_lock_trajectory.png",dpi=320,facecolor="white"); fig.savefig(OUTPUT/"Mouse_resolver_lock_trajectory.pdf",facecolor="white"); plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts",type=Path,default=COUNTS)
    parser.add_argument("--metadata",type=Path,default=META)
    parser.add_argument("--pooled-trajectories",type=Path,default=POOLED)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    return parser.parse_args()


def main():
    global COUNTS,META,POOLED,OUTPUT
    args=parse_args()
    COUNTS,META,POOLED,OUTPUT=args.counts,args.metadata,args.pooled_trajectories,args.output_dir
    OUTPUT.mkdir(parents=True,exist_ok=True)
    samples,audit=build_scores(); pooled=pooled_trajectories(); geom=geometry(samples)
    tests=pd.concat([exact_group_test(samples,outcome) for outcome in ["resolver_axis","lock_axis","control_imbalance"]],ignore_index=True)
    plot(samples,pooled,geom)
    samples.to_csv(OUTPUT/"GSE295293_mouse_phase_scores.tsv",sep="\t",index=False); audit.to_csv(OUTPUT/"mouse_module_mapping.tsv",sep="\t",index=False); tests.to_csv(OUTPUT/"GSE295293_exact_phase_tests.tsv",sep="\t",index=False); pooled.to_csv(OUTPUT/"GSE305331_cross_etiology_phase_trajectories.tsv",sep="\t",index=False)
    summary={"n_replicated_mice":len(samples),"geometry":geom,"tests":tests.to_dict(orient="records"),"claim":"Withdrawal follows a non-reciprocal trajectory with residual displacement in resolver-lock space."}
    (OUTPUT/"mouse_phase_trajectory_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n"); print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
