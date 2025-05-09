import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("🧪 Unsteady-State Multi-Species Reactor Simulator")
st.markdown("""
This simulator allows you to model unsteady-state behavior of chemical reactors (Batch, CSTR, PFR)
with support for multiple species, reversible reactions, custom rate laws, and jacket heat transfer.
""")

# ---- Species Setup ----
st.sidebar.header("1. Species Setup")
n_species = st.sidebar.number_input("Number of Species", min_value=1, value=2, step=1)
species_labels = [st.sidebar.text_input(f"Species {i+1} label", value=f"A{i+1}") for i in range(n_species)]

# ---- Reaction Setup ----
st.sidebar.header("2. Reaction Setup")
n_reactions = st.sidebar.number_input("Number of Reactions", min_value=1, value=1, step=1)
reactions = []
Ea = st.sidebar.number_input("Activation Energy (J/mol)", value=500.0)
R=8.314
Tf = st.sidebar.number_input("Feed Temperature Tf (K)", value=350.0)
T0 = st.sidebar.number_input("Initial Reactor Temp T0 (K)", value=300.0)
for r in range(n_reactions):
    with st.sidebar.expander(f"Reaction {r+1}"):
        reversible = st.checkbox(f"Reversible Reaction {r+1}?", value=True, key=f"rev_{r}")
        k_fwd = st.number_input(f"k_fwd for Reaction {r+1}", value=1.0, key=f"kf_{r}")
        k_bwd = st.number_input(f"k_bwd for Reaction {r+1}", value=1.0, key=f"kb_{r}") if reversible else 0.0
        fwd_expr = st.text_input(f"Forward rate expression for Reaction {r+1} (optional)", key=f"fexpr_{r}")
        bwd_expr = st.text_input(f"Backward rate expression for Reaction {r+1} (optional)", key=f"bexpr_{r}") if reversible else None
        
        st.markdown("**Stoichiometry (Reactants: -ve, Products: +ve)**")
        stoich = {}
        coeff=[]
        Cp = []
        for i, label in enumerate(species_labels):
            coef = st.number_input(f"Stoichiometric coefficient for {label} in R{r+1}", value=1, key=f"st_{r}_{i}")
            coeff.append(coef)
            cp = st.number_input(f"Heat Capacity (C_p) for Reactant {i + 1} (J/mol·K)", value=100.0)
            Cp.append(cp)
            stoich[label] = coef

        reactions.append({
            "reversible": reversible,
            "k_fwd": k_fwd,
            "k_bwd": k_bwd,
            "fwd_expr": fwd_expr,
            "bwd_expr": bwd_expr,
            "stoich": stoich
        })

# ---- Reactor Configuration ----
st.sidebar.header("3. Reactor Configuration")
reactor_type = st.sidebar.selectbox("Reactor Type", ["Batch", "CSTR", "PFR"])
V = st.sidebar.number_input("Volume (m^3)", value=1.0)
flow_rate = st.sidebar.number_input("Flow rate (for CSTR/PFR) (m^3/s)", value=1.0)

# ---- Initial Conditions ----
st.sidebar.header("4. Initial Conditions")
initial_conc = [st.sidebar.number_input(f"Initial {label} (mol/m^3)", value=1.0) for label in species_labels]


# ---- Heat Transfer ----
st.sidebar.header("5. Jacket Heat Transfer")
U = st.sidebar.number_input("Heat transfer coefficient U (W/m^2-K)", value=100.0)
A = st.sidebar.number_input("Heat transfer area A (m^2)", value=10.0)
Ta1 = st.sidebar.number_input("Coolant temperature Ta (K)", value=298.0)
mc=st.sidebar.number_input("Mass flow rate for coolant", value=0.1)   
Cpc = st.sidebar.number_input(f"Heat Capacity (C_p) for coolant (J/mol·K)", value=100.0)
T_ref=st.sidebar.number_input(f"Reference Temperature for Standard Heat of Reaction (K)", value=298)
dH_std = st.sidebar.number_input(f"Standard Heat of reaction at {T_ref} K (J/mol)", value=100.0)

# ---- Rate Law Generator ----
def rate_laws(conc, T):
    rate_list = []
    local_vars = {label: float(conc[i]) for i, label in enumerate(species_labels)}
    local_vars['T'] = T
    for rxn in reactions:
        if rxn['fwd_expr']:
            rate_fwd = eval(rxn['fwd_expr'], {}, local_vars) *np.exp(Ea/R*(1/Tf-1/T))
       
        else:
            rate_fwd = rxn['k_fwd'] * np.prod([conc[i] ** (-rxn['stoich'][label])
                                               for i, label in enumerate(species_labels)
                                               if rxn['stoich'][label] < 0])
        rate_total = rate_fwd
        if rxn['reversible']:
            if rxn['bwd_expr']:
                rate_bwd = eval(rxn['bwd_expr'], {}, local_vars) * np.exp(Ea/R*(1/Tf-1/T))
            else:
                rate_bwd = rxn['k_bwd'] * np.prod([conc[i] ** (rxn['stoich'][label])
                                                   for i, label in enumerate(species_labels)
                                                   if rxn['stoich'][label] > 0])
            rate_total -= rate_bwd
        rate_list.append(rate_total)
    return rate_list

# ---- ODE Function ----
def odes(t, y):
    C = y[:n_species]
    T = y[-1]
    X=(C[0]-initial_conc[0])/initial_conc[0]
    r = rate_laws(C, T)
    CpS=0
    for i in range(n_species):
        CpS+=Cp[i]*(initial_conc[i]/initial_conc[0])
    for i in range(n_species):
        CpS+=initial_conc[i]*X*coeff[i]
    
    dCdt = np.zeros(n_species)
    
    Ta2=T-(T-Ta1)*np.exp(-U*A/(mc*Cpc))
    Qr = mc*Cpc*(T-Ta1)*(1-np.exp(-U*A/mc/Cpc))
    Qg=0
    dH = dH_std + sum([coeff[i] * Cp[i] * (T - T_ref) for i in range(n_species)])
    for i in range(len(reactions)):
        Qg+=rate_laws(initial_conc, T)[i]*(dH)*V
    for j, rxn in enumerate(reactions):
        for i, label in enumerate(species_labels):
            dCdt[i] += rxn['stoich'][label] * r[j]
   
    if reactor_type == "CSTR":
        for i in range(n_species):
            dCdt[i] += (flow_rate / V) * (initial_conc[i] - C[i])
            
        dTdt = (flow_rate * Cp[i] * (T-T0) +Qg-Qr) / CpS
    elif reactor_type == "PFR":
        # PFR modeled with time as space proxy, for simplicity
        dTdt = (Qg-Qr) / CpS
    else:  # Batch
        dTdt = (Qg-Qr) / CpS

    return list(dCdt) + [dTdt]

# ---- Solve ----
time_span = st.sidebar.number_input("Enter time span: ")

if st.button("Run Simulation"):
    y0 = initial_conc + [T0]
    sol = solve_ivp(odes, [0, time_span], y0, method='RK45', dense_output=True)

    t_vals = np.linspace(*[0, time_span], 300)
    y_vals = sol.sol(t_vals)

    st.subheader("Concentration Profiles")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(species_labels):
        ax.plot(t_vals, y_vals[i], label=label)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Concentration (mol/L)")
    ax.set_title(f"Concentration Profiles in {reactor_type}")
    ax.legend()
    st.pyplot(fig)
    st.subheader("Temperature Profile")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(t_vals, y_vals[-1], label=label)
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Temperature (K)")
    ax1.set_title(f"Temperature Profile in {reactor_type}")
    st.pyplot(fig1)
