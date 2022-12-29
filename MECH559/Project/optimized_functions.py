import os
from DMDO import *
import math
import numpy as np
from numpy import cos, exp, pi, prod, sin, sqrt, subtract, inf
import copy
from typing import List, Dict, Any, Callable, Protocol, Optional

from blackbox import structure, thermal

user = USER

def get_inputs(V:List[variableData], names:List[str], sp_index:int):
  """This function returns a list of variable objects based on name and subproblem index"""
  variables = [vi for vi in V if vi.name in names and vi.sp_index==sp_index and vi.coupling_type not in [COUPLING_TYPE.FEEDFORWARD]]
  ordered_variables = []
  for name in names:
    for vi in variables:
      if vi.name == name:
        ordered_variables += [vi]
  return ordered_variables

def get_outputs(V:List[variableData], names:List[str], sp_index:int):
  """This function returns a list of variable objects based on name and subproblem index"""
  variables = [vi for vi in V if vi.name in names and vi.sp_index==sp_index and vi.coupling_type in [COUPLING_TYPE.FEEDFORWARD]]
  ordered_variables = []
  for name in names:
    for vi in variables:
      if vi.name == name:
        ordered_variables += [vi]
  return ordered_variables

def structure_opt(x, y):

  inputs = {}
  for input_name,value in zip(["r_rotor","r_0","ri","t","r_c", "r","tol","r_k"],x): # these are your targets
    inputs[input_name] = value

  outputs = {}
  
  for outputs_name,value in zip(["stress", "dist"],y): # these are your responses
    outputs[outputs_name] = value

  # Here I would define some of the constraints regrading the system, need to add constraints for radius
  G1 = outputs["stress"] - user.SIGMA
  G2 = outputs["tol"] - user.TOL
  G3 = input["r_0"] - input["r_rotor"]
  G4 = input["r_k"] - input["ri"]
  G5 = input["ri"] - input["r_0"]
  G6 = input["r_k"] - input["r_rotor"]
  G7 = 16.95 - input["t"]
  G8 = input["t"] - 17.05
  return [outputs["dist"], [G1,G2,G3,G4,G5,G6,G7,G8]] # one objective, seven constraint

def thermal_opt(x, y):
  # These is thermal funtion where system needs to fit into tolerences defined by structural team
  inputs = {}
  for input_name,value in zip(["r_rotor","t", "r_0", "r_i","r_c", "r"],x): # these are your targets
    inputs[input_name] = value

  outputs = {}
  for outputs_name,value in zip(["tol","del_T"],y): # these are your responses
    outputs[outputs_name] = value

  g1 = outputs["del_T"] - 316
  g2 = input["ri"] - input["r_0"]

  return[0, [g1,g2]] # no objective and one constraint

def Rotor_design():
  """ Here is the desing of a rotor"""
  # Variables definition
  s  = COUPLING_TYPE.SHARED
  ff = COUPLING_TYPE.FEEDFORWARD
  fb = COUPLING_TYPE.FEEDBACK
  un = COUPLING_TYPE.UNCOUPLED
  dum = COUPLING_TYPE.DUMMY
  v = {
    #subproblem 1: (structural Analysis)
    #shared variables:
    "var1":     {"index": 1     , "sp_index": 1 , f"name": "r_0"        , "coupling_type": s   , "link": 2     , "lb": 0.099 , "ub": 0.1255 , "baseline": 1.0},   
    "var2":     {"index": 2     , "sp_index": 1 , f"name": "r_i"        , "coupling_type": s   , "link": 2     , "lb": 0.07  , "ub": 0.099  , "baseline": 1.0},      
     #feedforward variables
    "var3":     {"index": 4     , "sp_index": 1 , f"name": "r_c"        , "coupling_type": ff   ,  "link": 2      , "lb": 0.038   , "ub": 0.042   , "baseline": 1.0},   
    "var4":     {"index": 5     , "sp_index": 1 , f"name": "r_rotor"    , "coupling_type": ff   ,  "link": 2      , "lb": 0.07     , "ub": 0.1317  , "baseline": 1.0},   
    "var5":     {"index": 6     , "sp_index": 1 , f"name": "r"          , "coupling_type": ff   ,  "link": 2      ,"lb": 0.00722  , "ub": 0.00798 , "baseline": 1.0}, 
    "var6":     {"index": 7     , "sp_index": 1 , f"name": "t"          , "coupling_type": s    ,   "link": 2      , "lb":  0.01695 , "ub": 0.01705 , "baseline": 1.0},
    #feedback variables
    "var7":     {"index": 8     , "sp_index": 1 , f"name": "tol"        , "coupling_type": fb   , "link": 2     , "lb": -0.05   , "ub": 0.05   , "baseline": 1.0},   
    #local variables
    "var8":     {"index": 9     , "sp_index": 1 , f"name": "r_k"        , "coupling_type": un   , "link": None  , "lb": 0.06676  , "ub": 0.0737   , "baseline": 1.0},   
     ## subproblem 2: (thermal analysis)
     #shared variables
    "var9":    {"index": 10    , "sp_index": 1 , f"name": "r_0"        , "coupling_type": s   , "link": 1     , "lb": 0.099 , "ub": 0.1255    , "baseline": 1.0}, 
    "var10":    {"index": 11    , "sp_index": 1 , f"name": "r_i"        , "coupling_type": s  ,  "link": 1     , "lb": 0.07  , "ub": 0.099     , "baseline": 1.0},
    
    #local varibles
    "var11":    {"index": 13    , "sp_index": 2 , f"name": "del_T"       , "coupling_type":un  , "link": None     , "lb": 1     , "ub": 316     , "baseline": 1.0},   
    #feedback variables:
    "var12":    {"index": 14    , "sp_index": 2 , f"name": "r_c"      , "coupling_type": fb   , "link": 1     , "lb": 0.038    , "ub": 0.042    , "baseline": 1.0},   
    "var13":    {"index": 15    , "sp_index": 2 , f"name": "r_rotor"  , "coupling_type": fb   , "link": 1     , "lb": 0.07     , "ub": 0.1317   , "baseline": 1.0},   
    "var14":    {"index": 16    , "sp_index": 2 , f"name": "r"        , "coupling_type": fb   , "link": 1     , "lb": 0.00722  , "ub": 0.00798  , "baseline": 1.0},   
    "var15":    {"index": 17    , "sp_index": 2 , f"name": "t"        , "coupling_type": s   , "link": 1     , "lb":  0.01695 , "ub": 0.01705  , "baseline": 1.0},  
    #feedforward variables:
    "var16":    {"index": 18    , "sp_index": 2 , f"name": "tol"      , "coupling_type": ff   , "link": 1     , "lb": -0.05    , "ub": 0.05     , "baseline": 1.0},   
    }

  Qscaling = []
  for key,value in v.items():
    scaling = v[key]["ub"] - v[key]["lb"]
    v[key]["scaling"] = scaling
    v[key]["dim"] = 1
    v[key]["value"] = v[key]["baseline"]
    Qscaling.append(.1/scaling if scaling != 0.0 and scaling != 0.0 else 1.)

  V: List[variableData] = []
  for i in range(len(v)):
    V.append(variableData(**v[f"var{i+1}"]))

  # Analyses setup; construct disciplinary analyses
  X1 = get_inputs(V,["r_rotor","r_0","ri","t","r_c", "r","tol","r_k"],sp_index=1)
  Y1 = get_outputs(V,["stress","dist"],sp_index=1)
  DA1: process = DA(inputs=X1,  
  outputs=Y1,
  blackbox=structure,
  links=[2],
  coupling_type=[COUPLING_TYPE.FEEDFORWARD])
  
  X2 = get_inputs(V,["r_rotor","t", "r_0", "r_i","r_c", "r"],sp_index=2) # inputs
  Y2 = get_outputs(V,["tol", "del_T"],sp_index=2)
  DA2: process = DA(inputs=X2,
  outputs=Y2,
  blackbox=thermal,
  links=[1],
  coupling_type=[COUPLING_TYPE.FEEDFORWARD]
  )


  # MDA setup; construct subproblems MDA
  sp1_MDA: process = MDA(nAnalyses=1, analyses = [DA1], variables=X1, responses=Y1)
  sp2_MDA: process = MDA(nAnalyses=1, analyses = [DA2], variables=X2, responses=Y2)
  # Construct the coordinator
  coord = ADMM(beta = 1.3,
  nsp=4,
  budget = 500,
  index_of_master_SP=1,
  display = True,
  scaling = Qscaling,
  mode = "serial",
  M_update_scheme= w_scheme.MEDIAN,
  store_q_io=True
  )

  # Construct subproblems
  V1 = get_inputs(V,["r_rotor","r_0","ri","t","r_c", "r","tol","r_k"],sp_index=1) # inputs
  R1 = get_outputs(V,["stress","dist",],sp_index=1)

  sp1 = SubProblem(nv = len(V1),
  index = 1,
  vars = V1,
  resps = R1,
  is_main=1,
  analysis= sp1_MDA,
  coordination=coord,
  opt=structure_opt,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  freal=30000)

  V2 = get_inputs(V,["r_rotor","t", "r_0", "r_i","r_c", "r"],sp_index=2) # inputs
  R2 = get_outputs(V,["tol", "del_T"],sp_index=2)

  sp2 = SubProblem(nv = len(V2),
  index = 2,
  vars = V2,
  resps = R2,
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=thermal_opt,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST
  )

  # Construct MDO workflow
  MDAO: MDO = MDO(
  Architecture = MDO_ARCHITECTURE.IDF,
  Coordinator = coord,
  subProblems = [sp1, sp2],
  variables = copy.deepcopy(V),
  responses = R1+R2,
  fmin = np.inf,
  hmin = np.inf,
  display = True,
  inc_stop = 1E-9,
  stop = "Iteration budget exhausted",
  tab_inc = [],
  noprogress_stop = 500
  )

  user = USER
#define constants here
  user = USER
#define constants here
  user.DENSITY = 7250 
  user.mu = 0.4
  user.PRESSURE = 1194000
  user.TOL = 0.05
  user.alpha = 0.000011
  user.SIGMA = 200000000
  user.poison = 0.3
  user.VELOCITY = 50
  user.THETA = math.pi*.25
  user.H = 10
  return MDAO
  
if __name__ == "__main__":
  MDAO = Rotor_design()
  out = MDAO.run()
  print(out)
