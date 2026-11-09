"""Kansas CO2/brine: genuine FVM runs with and without the approved branching barriers.

Smooth geometric plane overlays are distinct from the staircase face multipliers
used in the equations. No throw, no altered properties/wells, and no FEM panels.
"""
from __future__ import annotations
import argparse, hashlib, json, time, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from kansas_window import kansas_structural, section_wells, background_args
from kansas_fluids import well_list, wells_for_plot, LAYERS
from kansas_branching_geometry import construct, plane_z, plane_dip, interp_top, MAIN_NW, MAIN_SE, SPLAY_SE
import platform, subprocess
from subsurfix import _native as nat

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if hasattr(x,"item") else x.tolist()))

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def human_duration(seconds):
    hours, rem = divmod(float(seconds), 3600)
    minutes, sec = divmod(rem, 60)
    return (f"{int(hours)} h " if hours else "") + (f"{int(minutes)} min " if minutes else "") + f"{sec:.1f} s"

def timing_setup():
    info = dict(os=platform.system(), architecture=platform.machine(),host=platform.node(),
        python_version=platform.python_version(), environment=Path(sys.executable).parent.parent.name,
        thread_cap=int(nat.flow_threads()),
        scope="native solver call wall clock; includes constitutive table/solver initialization; excludes shared mesh construction, cache I/O and rendering")
    if platform.system()=="Darwin":
        info["cpu"]=subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"],text=True).strip()
        info["memory_GiB"]=int(subprocess.check_output(["sysctl","-n","hw.memsize"],text=True))/2**30
    return info
YR = 365.25 * 86400
TITLES = ["FVM: pressure — no faults", "FVM: CO2 saturation — no faults",
          "FVM: pressure — two faults", "FVM: CO2 saturation — two faults"]
CAPTION = ("The FVM results use the FIM-OBL solver. Both rows use the Kansas structural model with the same wells "
    "and CO₂-into-brine setup; the lower row adds two no-throw branching staircase barriers "
    "(main about 4° SW; branch about 5° NE soling into the main; multiplier 10⁻⁴). "
    "The gentle dips follow the approved traces. Figures draw the geometric planes smoothly, not the staircase faces. "
    "This shows pressure and saturation, not temperature. FEM is omitted because faulted multiphase FEM is not available.")

def setup(a):
    bm = kansas_structural()
    nx, ny, nz = bm["dims"]
    plain = nat.FlowModel.from_corner_point(bm["grid"], bm["perm"], bm["phi"])
    xyz = np.asarray(plain.centers()).reshape(nz, ny, nx, 3)
    geom = construct(bm["xs"], bm["ys"], bm["top_nodes"])
    inj, prod = tuple(getattr(a,"inj",(5,10))), tuple(getattr(a,"prod",(9,16)))
    inj_layers = list(getattr(a,"inj_layers",range(8,12)))
    prod_layers = list(getattr(a,"prod_layers",range(1,4)))
    rate = float(getattr(a,"rate",.008)); bhp = float(getattr(a,"prod_bhp",1.2e7))
    p_range = tuple(getattr(a,"p_range",(5e6,2e8)))
    table_points = list(getattr(a,"table_points",(261,201)))
    if not (rate > 0 and 0 < p_range[0] < p_range[1] and len(table_points)==2 and min(table_points)>=2):
        raise ValueError("Invalid injection rate or constitutive table settings")
    if not inj_layers or not prod_layers or min(inj_layers+prod_layers)<0 or max(inj_layers+prod_layers)>=nz:
        raise ValueError("Open layers must be non-empty subsets of the model layers")
    LAYERS.update(injector=inj_layers, producer=prod_layers)
    wells = well_list(bm, [inj], [prod], rate, bhp)
    times = np.linspace(a.years/a.frames, a.years, a.frames)
    params = dict(model="kansas_structural", dims=list(bm["dims"]), cell=list(bm["cell"]),
        active_cells=int(bm["active"].sum()), notch=[14,11], rate_m3_s=rate,
        inj_column=list(inj),prod_column=list(prod),inj_layers=inj_layers,prod_layers=prod_layers,
        n_threads=int(getattr(a,"n_threads",8)),
        report_years=times.tolist(), wells=wells, initial_pressure_Pa=2e7, temperature_K=323.15,
        m_nacl=1.0, gravity=9.80665, p_range_Pa=list(p_range), table_points=table_points,
        dt_init_s=86400., dt_max_s=.25*YR, multiplier=a.multiplier,
        geometry={k:{j:np.asarray(v).tolist() for j,v in geom[k].items()} for k in ["main","splay"]},
        construction_points={k:geom[k].tolist() for k in ["B","N","S","STEM","J"]},
        property_sha256={k:hashlib.sha256(np.asarray(bm[k]).tobytes()).hexdigest() for k in ["perm","phi","active","top_nodes"]})
    fingerprint=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()
    return dict(bm=bm,plain=plain,xyz=xyz,geom=geom,wells=wells,times=times,params=params,fingerprint=fingerprint)

def in_reach(g, name, hit):
    if name=="main": start, v, lo, hi=MAIN_NW, MAIN_SE-MAIN_NW, -.05, 1.08
    else: start, v, lo, hi=g["B"][:2], SPLAY_SE-g["B"][:2], -.02, 1.06
    t=(hit[...,:2]-start)@v/(v@v)
    return (t>=lo)&(t<=hi)

def models(b,a):
    xyz=b["xyz"];shape=xyz.shape[:-1];active=b["bm"]["active"].reshape(shape);g=b["geom"]
    mult={key:np.ones(shape) for key in ["MULTX","MULTY","MULTZ"]}
    counts={name:{} for name in ["main","splay"]};hits={name:[] for name in counts}
    for axis,key in [(2,"MULTX"),(1,"MULTY"),(0,"MULTZ")]:
        sl=[slice(None)]*3;sr=sl.copy();sl[axis]=slice(None,-1);sr[axis]=slice(1,None);sl,sr=tuple(sl),tuple(sr)
        q,r=xyz[sl],xyz[sr]
        for name in counts:
            pl=g[name];d0=(q-pl["p0"])@pl["n"];d1=(r-pl["p0"])@pl["n"]
            frac=np.divide(d0,d0-d1,out=np.zeros_like(d0),where=abs(d0-d1)>1e-14)
            hit=q+frac[...,None]*(r-q)
            cross=((d0>=0)!=(d1>=0)) & active[sl] & active[sr] & in_reach(g,name,hit)
            if name=="splay": cross &= hit[...,2]<=plane_z(g["main"],hit[...,0],hit[...,1])+1e-9
            mult[key][sl][cross]=a.multiplier
            counts[name][key]=int(cross.sum());hits[name].append(hit[cross])
    fault=nat.FlowModel.from_corner_point(b["bm"]["grid"],b["bm"]["perm"],b["bm"]["phi"],mult={k:v.ravel() for k,v in mult.items()})
    ia,ib,t0=b["plain"].connections();ja,jb,t1=fault.connections()
    assert np.array_equal(ia,ja) and np.array_equal(ib,jb)
    changed=~np.isclose(t1/t0,1,rtol=1e-10,atol=0)
    assert changed.sum()==sum(int((v<1).sum()) for v in mult.values())>0
    assert np.allclose((t1/t0)[changed],a.multiplier,rtol=1e-10,atol=0)
    assert np.array_equal(b["plain"].pore_volume(),fault.pore_volume())
    assert fault.n_fault_connections==b["plain"].n_fault_connections==0
    h=np.concatenate(hits["splay"]);assert len(h)>0
    gap=plane_z(g["main"],h[:,0],h[:,1])-h[:,2];assert gap.min()>=-1e-8
    meta=dict(face_counts=counts,changed_connections=int(changed.sum()),connections=len(t0),
        applied_multiplier=a.multiplier,no_throw=True,splay_min_depth_above_main_m=float(gap.min()),
        plane_dips={k:list(plane_dip(g[k])) for k in ["main","splay"]},
        formulation="Neighbour-centre plane crossings; splay hits clipped shallower than main; finite preview reaches; shared-face multiplier once, not compounded.")
    return dict(no_faults=b["plain"],two_faults=fault),meta

def solve(name,model,b,a):
    path=a.cache_dir/(name+".npz")
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            assert str(z["fingerprint"])==b["fingerprint"],"cache geometry/setup mismatch"
            return {k:z[k].copy() for k in z.files}
    n=b["bm"]["phi"].size
    print("Starting",name,n,"cells",nat.flow_threads(),"threads",flush=True)
    t=time.perf_counter()
    try:
        r=nat.run_fim_co2_brine(model,b["wells"],np.full(n,2e7),np.zeros(n),[0.]+(b["times"]*YR).tolist(),
            temperature=323.15,m_nacl=1.,gravity=9.80665,p_range=tuple(b["params"]["p_range_Pa"]),table_points=b["params"]["table_points"],
            dt_init=86400.,dt_max=.25*YR,linear_solver="iterative")
    except Exception as error:
        elapsed=time.perf_counter()-t
        dump(a.cache_dir/(name+"_failure.json"),dict(status="failed",error=str(error),
            failed_attempt_wall_clock_seconds=elapsed,runtime_setup=timing_setup(),fingerprint=b["fingerprint"]))
        raise
    runtime=time.perf_counter()-t;act=b["bm"]["active"]
    P=np.asarray(r["p"])[1:]/1e6;S=np.asarray(r["saturation"])[1:,:,1]
    mip,net=np.asarray(r["mass_in_place"]),np.asarray(r["mass_net_in"])
    err=(mip[-1]-mip[0])-(net[-1]-net[0]);den=np.maximum(abs(net[-1]-net[0]),1e-12*np.maximum(abs(mip[0]),1))
    pc=b["wells"][1]["cells"]
    stats=dict(runtime_s=runtime,wall_clock_seconds=runtime,wall_clock_human=human_duration(runtime),runtime_setup=timing_setup(),
        component_order=["CO2","H2O"],mass_balance_error_kg=err.tolist(),mass_balance_relative_to_net_transfer=(abs(err)/den).tolist(),
        mass_balance_relative_initial_total=float(abs(err).max()/abs(mip[0]).sum()),
        pressure_min_MPa=float(P[:,act].min()),pressure_max_MPa=float(P[:,act].max()),
        producer_cell_pressure_final_MPa=float(P[-1,pc].mean()),producer_CO2_saturation_final=float(S[-1,pc].max()),
        producer_CO2_saturation_max=float(S[:,pc].max()),max_CO2_saturation=float(S[:,act].max()),
        final_cells_saturation_gt_0_01=int((S[-1,act]>.01).sum()),injector_bhp_max_MPa=float(np.asarray(r["bhp"])[:,0].max()/1e6),
        producer_bhp_final_MPa=float(np.asarray(r["bhp"])[-1,1]/1e6),
        **{k:int(r.get(k,0)) for k in ["n_steps","n_cuts","n_newton","n_linear_iterations","n_linear_fallbacks","n_fallbacks"]})
    dump(a.cache_dir/(name+"_stats.json"),stats)  # persist before any scientific rejection
    assert P.shape==(len(b["times"]),n) and np.isfinite(P[:,act]).all() and np.isfinite(S[:,act]).all()
    assert P[:,act].min()>=b["params"]["p_range_Pa"][0]/1e6 and P[:,act].max()<=b["params"]["p_range_Pa"][1]/1e6,"pressure outside constitutive table range"
    assert S[:,act].min()>=-1e-9 and S[:,act].max()<=1+1e-9
    if max(stats["mass_balance_relative_to_net_transfer"])>=1e-5 or stats["n_fallbacks"] or stats["injector_bhp_max_MPa"]>=60:
        dump(a.cache_dir/(name+"_validation_failure.json"),dict(status="rejected",stats=stats,
            fingerprint=b["fingerprint"],reason="Mass-balance tolerance or constitutive-property fallback check failed."))
    assert max(stats["mass_balance_relative_to_net_transfer"])<1e-5,"mass balance failed"
    assert stats["n_fallbacks"]==0,"constitutive fallback occurred"
    assert stats["injector_bhp_max_MPa"]<60,"injector BHP reached the 60 MPa flash-validity ceiling"
    out=dict(P=P,S=S,times=b["times"],stats=np.array(json.dumps(stats)),fingerprint=np.array(b["fingerprint"]),mass_in_place=mip,mass_net_in=net,bhp=np.asarray(r["bhp"]))
    np.savez_compressed(path,**out);print(name,json.dumps(stats),flush=True);return out

def smooth_surfaces(b):
    import pyvista as pv
    bm,g=b["bm"],b["geom"];xs,ys=bm["xs"],bm["ys"]
    x=np.linspace(xs[0],xs[-1],177);y=np.linspace(ys[0],ys[-1],177);X,Y=np.meshgrid(x,y,indexing="ij")
    top=interp_top(xs,ys,bm["top_nodes"],X.ravel(),Y.ravel()).reshape(X.shape)
    surfaces=[]
    for name,color in [("main","#C2185B"),("splay","#E040A0")]:
        Z=plane_z(g[name],X,Y);mesh=pv.StructuredGrid(X,Y,-Z)
        mesh["upper"]=(Z-top).ravel(order="F");mesh["lower"]=(top+60-Z).ravel(order="F")
        start,v,lo,hi=(MAIN_NW,MAIN_SE-MAIN_NW,-.05,1.08) if name=="main" else (g["B"][:2],SPLAY_SE-g["B"][:2],-.02,1.06)
        t=((np.stack([X,Y],axis=-1)-start)@v)/(v@v)
        mesh["reach0"]=(t-lo).ravel(order="F");mesh["reach1"]=(hi-t).ravel(order="F")
        if name=="splay":mesh["sole"]=(plane_z(g["main"],X,Y)-Z).ravel(order="F")
        for key in ["upper","lower","reach0","reach1"]+(["sole"] if name=="splay" else []):mesh=mesh.clip_scalar(scalars=key,value=0,invert=False)
        # Keep the same L-shaped support; two non-overlapping rectangles avoid drawing through the notch.
        left=mesh.clip(normal="x",origin=(xs[14],0,0),invert=True)
        right=mesh.clip(normal="x",origin=(xs[14],0,0),invert=False).clip(normal="y",origin=(0,ys[12],0),invert=False)
        surfaces.append((left.merge(right).extract_surface(algorithm="dataset_surface"),color))
    dist=np.linspace(-1500,g["length"]+900,1600);xy=g["o"]+dist[:,None]*g["u"]
    top=interp_top(xs,ys,bm["top_nodes"],xy[:,0],xy[:,1]);lines=[]
    for name,color in [("main","#C2185B"),("splay","#E040A0")]:
        z=plane_z(g[name],xy[:,0],xy[:,1]);hit=np.c_[xy,z]
        keep=(z>=top)&(z<=top+60)&in_reach(g,name,hit)
        if name=="splay":keep &= z<=plane_z(g["main"],xy[:,0],xy[:,1])+1e-9
        lines.append((np.c_[dist[keep]/1000,z[keep]],color))
    return surfaces,lines

def render(b,cases,a):
    from subsurfix.viz3d import LayeredGrid,animate_layered
    bm=b["bm"];grid=LayeredGrid(bm["xs"],bm["ys"],bm["top_nodes"],12,5,bm["active"])
    inj,prod=tuple(b["params"]["inj_column"]),tuple(b["params"]["prod_column"])
    wl=wells_for_plot(bm,grid,[inj],[prod])
    sec=section_wells(bm,inj,prod,wl[0].extra["open"],wl[1].extra["open"]);sec["height"]=210
    surfaces,lines=smooth_surfaces(b)
    lo=20.;hi=max(float(c["P"][:,bm["active"]].max()) for c in cases.values())
    panels=[]
    for row,name in enumerate(["no_faults","two_faults"]):
        for col,key in enumerate(["P","S"]):
            opts=dict(label="Pressure" if key=="P" else "CO₂ saturation",units="MPa" if key=="P" else "",clim=(lo,hi) if key=="P" else (0,1),
                cmap="viridis_bright",bar_fmt="%.1f",ref=20 if key=="P" else 0,
                plume=.02,section_alpha=.85)
            if key=="P":opts["positive_change_only"]=True
            if row:opts.update(fault_surfaces=surfaces,fault_section=lines,fault_opacity=.65)
            panels.append((TITLES[2*row+col],cases[name][key],opts))
    animate_layered(grid,panels,a.gif,times=b["times"],wells=wl,shape=(2,2),panel_size=(640,360),section=sec,
        background=bm["k_md"],background_clim=(1,1000),background_log=True,background_label="permeability (mD)",
        context_opacity=.18,rotate=bool(getattr(a,"rotate",True)),rotate_deg=float(getattr(a,"rotate_deg",35.)),view_dir=tuple(float(x) for x in getattr(a,"view_dir",(.88,-.60,.75))),hold_last=0,still=a.still,
        header=f"Mississippian interval, Kansas · FVM CO2 into brine, {b['params']['rate_m3_s']:g} m³/s · no faults vs two branching faults",
        footer="Kansas 22 × 22 × 12; 4656 active cells; 11 × 11 km × 60 m; vertical exaggeration 40×; same KGS properties and wells.\n"
        f"INJ {inj}, layers {b['params']['inj_layers']}; PROD {prod}, layers {b['params']['prod_layers']}, {b['wells'][1]['bhp']/1e6:g} MPa BHP; 50 °C brine.\n"
        f"FVM results use the FIM-OBL solver; staircase multiplier {a.multiplier:g}; smooth pink planes; pressure and saturation, not temperature.\n"
        "Pressure colour: cell P > 20 MPa only; no-fault cells stay below 20 MPa (well BHP is distinct). Pink: smooth 3D planes and section traces.")
    from PIL import Image,ImageSequence
    im=Image.open(a.gif);frames=[f.copy() for f in ImageSequence.Iterator(im)];dur=[250]*len(frames);dur[-1]=1500
    frames[0].save(a.gif,save_all=True,append_images=frames[1:],duration=dur,loop=0,optimize=False)
    return dict(sha256=sha(a.gif),png_sha256=sha(a.still),size=list(im.size),frame_count=len(frames),frame_durations_ms=dur)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--cache-dir",type=Path,required=True)
    ap.add_argument("--rate",type=float,default=.008);ap.add_argument("--years",type=float,default=100);ap.add_argument("--frames",type=int,default=50);ap.add_argument("--multiplier",type=float,default=1e-4)
    ap.add_argument("--save-npz",type=Path);ap.add_argument("--from-npz",type=Path);ap.add_argument("--validate-only",action="store_true")
    ap.add_argument("--no-rotate",dest="rotate",action="store_false",help="fixed camera");ap.add_argument("--rotate-deg",type=float,default=35.,help="total camera azimuth sweep in degrees when rotating");ap.add_argument("--view-dir",type=float,nargs=3,default=[.88,-.60,.75],metavar=("X","Y","Z"),help="starting direction from model to camera (x east, y north, z up)");ap.add_argument("--gif",type=Path,default=ROOT/"image/faulted_co2_kansas.gif");ap.add_argument("--still",type=Path,default=ROOT/"image/faulted_co2_kansas.png")
    return _execute(ap.parse_args())


def _execute(a):
    a.cache_dir.mkdir(parents=True,exist_ok=True)
    n_threads=int(getattr(a,"n_threads",8))
    if not 1 <= n_threads <= 8: raise ValueError("thread cap must be between 1 and 8")
    nat.set_flow_threads(n_threads)
    b=setup(a);mm,meta=models(b,a);print("Geometry and native multiplier checks",json.dumps(meta),flush=True)
    if a.validate_only:return
    if a.from_npz:
        with np.load(a.from_npz,allow_pickle=False) as z:
            assert str(z["fingerprint"])==b["fingerprint"]
            cases={n:{k.split("|",1)[1]:z[k].copy() for k in z.files if k.startswith(n+"|")} for n in mm}
    else:cases={n:solve(n,m,b,a) for n,m in mm.items()}
    if not a.from_npz:
        np.savez_compressed(a.save_npz or a.cache_dir/"combined.npz",fingerprint=np.array(b["fingerprint"]),**{n+"|"+k:v for n,c in cases.items() for k,v in c.items()})
    stats={n:json.loads(str(c["stats"])) for n,c in cases.items()};dt=stats["two_faults"]["runtime_s"]-stats["no_faults"]["runtime_s"]
    result=dict(status="computed",version="0.2.0",case=b["params"],barriers=meta,cases=stats,titles=TITLES,caption=CAPTION.replace("10⁻⁴",f"{a.multiplier:g}"),
        fingerprint=b["fingerprint"],timing_difference_seconds=dt,timing_difference_human=f"{dt:+.1f} s",
        timing_note="One fresh sequential call per case on the same Mac; not a repeated-trial benchmark; native solver-call wall clock excludes common mesh setup, caches and rendering.",
        comparison=dict(max_final_saturation_difference=float(abs(cases["two_faults"]["S"][-1]-cases["no_faults"]["S"][-1]).max()),
            final_cells_saturation_difference_gt_0_01=int((abs(cases["two_faults"]["S"][-1]-cases["no_faults"]["S"][-1])>.01).sum())))
    report_path=a.cache_dir/("visual_refresh.json" if a.from_npz else "result.json")
    dump(report_path,result)
    if getattr(a,"render_output",True):
        result["artifact"]=render(b,cases,a);result["status"]="complete"
    dump(report_path,result)
    print(json.dumps(result),flush=True)
    return result


def run_comparison(*, cache_dir, rate=.008, p_range=(5e6,2e8), table_points=(261,201),
                   years=100, frames=50, fault_multiplier=1e-4, inj=(5,10), prod=(9,16),
                   inj_layers=tuple(range(8,12)), prod_layers=tuple(range(1,4)),
                   prod_bhp=12e6, n_threads=8, render_output=False, gif=None, still=None,
                   rotate=True, rotate_deg=35., view_dir=(.88,-.60,.75)):
    """Explicit notebook/API entry point; call only behind the notebook's opt-in flag.

    ``view_dir`` is the starting model-to-camera direction; ``rotate`` turns the camera sweep
    on/off; ``rotate_deg`` is its total degrees (35 from a SSE-leaning start to an almost-east end).
    Shares setup, native solver, cache fingerprint, scientific rejection checks and
    geometry with the CLI. Does not make a widened table physically valid.
    """
    a=argparse.Namespace(cache_dir=Path(cache_dir),rate=rate,p_range=tuple(p_range),
        table_points=list(table_points),years=years,frames=frames,multiplier=fault_multiplier,
        inj=tuple(inj),prod=tuple(prod),inj_layers=list(inj_layers),prod_layers=list(prod_layers),
        prod_bhp=prod_bhp,n_threads=n_threads,render_output=render_output,rotate=bool(rotate),rotate_deg=float(rotate_deg),view_dir=tuple(float(x) for x in view_dir),
        gif=Path(gif) if gif is not None else ROOT/"image/faulted_co2_kansas.gif",
        still=Path(still) if still is not None else ROOT/"image/faulted_co2_kansas.png",
        save_npz=None,from_npz=None,validate_only=False)
    return _execute(a)

if __name__=="__main__":main()
