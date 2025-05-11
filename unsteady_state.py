import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("Unsteady-State Multi-Species Reactor Simulator")

st.sidebar.header("1. Species Setup")
n_reactants = st.sidebar.number_input("Number of Reactants", min_value=1, value=1, step=1)
n_products = st.sidebar.number_input("Number of Products", min_value=1, value=1, step=1)
reactant_labels = [st.sidebar.text_input(f"Reactant {i+1} label", value=f"R{i+1}") for i in range(n_reactants)]
product_labels=[st.sidebar.text_input(f"Product {i+1} label", value=f"P{i+1}") for i in range(n_products)]
n_species=n_reactants+n_products
species_labels=reactant_labels+product_labels
st.sidebar.header("2. Reaction Setup")
n_reactions = st.sidebar.number_input("Number of Reactions in series", min_value=1, value=1, step=1)
reactions = []
Ea = st.sidebar.number_input("Activation Energy (J/mol)", value=500.0)
R=8.314
#Tf = st.sidebar.number_input("Feed Temperature Tf (K)", value=350.0)
T0 = st.sidebar.number_input("Initial Reactor Temp T0 (K)", value=300.0)
for r in range(n_reactions):
    with st.sidebar.expander(f"Reaction {r+1}"):
        reversible = st.checkbox(f"Reversible Reaction {r+1}?", value=True, key=f"rev_{r}")
        k_fwd = st.number_input(f"k_fwd for Reaction {r+1}", value=1.0, key=f"kf_{r}")
        k_bwd = st.number_input(f"k_bwd for Reaction {r+1}", value=0.0, key=f"kb_{r}") if reversible else 0.0
        fwd_expr = st.text_input(f"Forward rate expression for Reaction {r+1} (optional)", key=f"fexpr_{r}")
        bwd_expr = st.text_input(f"Backward rate expression for Reaction {r+1} (optional)", key=f"bexpr_{r}") if reversible else None
        
        st.markdown("Stoichiometry (Reactants: -ve, Products: +ve)")
        stoich = {}
        coeff=[]
        Cp = []
        
        for i, label in enumerate(reactant_labels):
            
            coef = st.number_input(f"Stoichiometric coefficient for {label} in R{r+1}", value=1, key=f"nu_r_{r}_{i}")
            coeff.append(-coef)
            cp = st.number_input(f"Heat Capacity (C_p) for Reactant {i + 1} (J/mol·K)", value=100.0, key=f"cp_r_{r}_{i}")
            Cp.append(cp)
            stoich[label] = -coef
        for j, label in enumerate(product_labels):
            
            coef = st.number_input(f"Stoichiometric coefficient for {label} in R{r+1}", value=1, key=f"nu_p_{r}_{j}")
            coeff.append(coef)
            cp = st.number_input(f"Heat Capacity (C_p) for Reactant {j + 1} (J/mol·K)", value=100.0, key=f"cp_p_{r}_{j}")
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

st.sidebar.header("3. Reactor Configuration")
reactor_type = st.sidebar.selectbox("Reactor Type", ["Batch", "CSTR", "PFR"])
v=0
rho=0
L=1
N=100
if reactor_type=="PFR":
    v=st.sidebar.number_input("Velocity (m/min): ", value=10)
    L=st.sidebar.number_input("Length of reactor (m): ", value=10)
    N=st.sidebar.number_input("Number of spatial segments: ", value=100)
V = st.sidebar.number_input("Volume (m^3)", value=1.0)
flow_rate = st.sidebar.number_input("Volumetric Flow rate (for CSTR/PFR) (m^3/s)", value=1.0)

st.sidebar.header("4. Initial Conditions")
initial_conc = [st.sidebar.number_input(f"Initial {label} (mol/m^3)", value=1.0, key=f"r_{label}") for label in reactant_labels]+[st.sidebar.number_input(f"Initial {label} (mol/m^3)", value=1.0, key=f"p_{label}") for label in product_labels]

st.sidebar.header("5. Jacket Heat Transfer")
U = st.sidebar.number_input("Heat transfer coefficient U (W/m^2-K)", value=10.0)
A = st.sidebar.number_input("Heat transfer area A (m^2)", value=10.0)
Ta1 = st.sidebar.number_input("Coolant temperature Ta (K)", value=298.0)
mc=st.sidebar.number_input("Mass flow rate for coolant", value=10)   
Cpc = st.sidebar.number_input(f"Heat Capacity (C_p) for coolant (J/mol·K)", value=10.0)
T_ref=st.sidebar.number_input(f"Reference Temperature for Standard Heat of Reaction (K)", value=290)
dH_std=[]
for i in range(n_reactions):
    dH_std.append(st.sidebar.number_input(f"Standard Heat of reaction {i} at {T_ref} K (J/mol)", value=-500.0, key=f"h_{i}"))

def rate_laws(conc, T):
    rate_list = [0.0 for _ in range(n_species)]  # initialize rate list for all species
    
    # Set up local variables for forward and backward rate expressions
    localr_vars = {label: float(conc[i]) for i, label in enumerate(reactant_labels)}
    localr_vars['T'] = T
    localp_vars = {label: float(conc[i]) for i, label in enumerate(product_labels)}
    localp_vars['T'] = T
    for rxn in reactions:
        # Forward rate
        if rxn['fwd_expr']:
            rate_fwd = eval(rxn['fwd_expr'], {}, localr_vars)
        else:
            rate_fwd = rxn['k_fwd']*np.exp((Ea/R) *(1/T0-1/T)) * np.prod([localr_vars[label] ** abs(rxn['stoich'].get(label, 0))
                                               for label in reactant_labels if label in rxn['stoich']])

        # Backward rate if reversible
        rate_bwd = 0.0
        if rxn['reversible']:
            if rxn['bwd_expr']:
                rate_bwd = eval(rxn['bwd_expr'], {}, localp_vars)
            else:
                rate_bwd = rxn['k_bwd'] *np.exp((Ea/R) *(1/T0-1/T))* np.prod([localp_vars[label] ** abs(rxn['stoich'].get(label, 0))
                                                   for label in product_labels if label in rxn['stoich']])
        
        rate_net = -rate_bwd + rate_fwd  # net rate of progress of the reaction

        # Accumulate species rates
        for i, species in enumerate(species_labels):
            nu_i = rxn['stoich'].get(species, 0)  # stoichiometric coefficient
            rate_list[i] += nu_i * rate_net  # accumulation

    return rate_list
    
def pfr_odes(t, y):
    C = y[:N]
    T = y[N:]
    Qr = 0
    Ta2 = T
    CpS = sum([Cp[i] * C[i] for i in range(n_species)])

    if mc != 0 and Cpc != 0:
        Ta2 = T - (T - Ta1) * np.exp(-U*A / (mc * Cpc))
        Qr = mc * Cpc * (T - Ta2) * (1 - np.exp(-U*A / (mc * Cpc)))

    deltaH = sum([coeff[i] * Cp[i] * (T - T_ref) for i in range(n_species)])
    dH = [dH_std[i] + deltaH for i in range(n_reactions)]
    rate_array = rate_laws(C, T)
    Qg = sum([rate_array[i] * dH[i] for i in range(n_reactions)])
    
    dCdt = np.zeros_like(C)
    dTdt = np.zeros_like(T)
    dz = L / N

    for i in range(1, len(T)):
        if(mc!=0 and Cpc!=0):
            dTdt[i] = -v * (T[i] - T[i-1]) / dz + (Qg[i] - Qr[i]) / CpS 
        else:
            dTdt[i] = -v * (T[i] - T[i-1]) / dz + (Qg[i]) / CpS 

    for j in range(1, min(len(rate_array), len(C))):
        dCdt[j] = -v * (C[j] - C[j-1]) / dz + rate_array[j][j-1]

    dCdt[0] = rate_array[0][0]
    dTdt[0] = 0
    T = np.clip(T, 100, 5000)
    C = np.clip(T, 0, None)
    return np.concatenate((dCdt, dTdt))

def odes(t, y):
    C = y[:n_species]
    T = y[-1]
    
    X=(C[0]-initial_conc[0])/initial_conc[0]
    r = rate_laws(C, T)
    CpS=0
    for i in range(n_species):
        CpS+=Cp[i]*(initial_conc[i])*V
    for i in range(n_species):
        CpS+=initial_conc[i]*X*coeff[i]
    
    dCdt = np.zeros(n_species)
    dTdt = np.zeros(n_species)
    Qr=0
    Ta2=T
    if mc!=0 and Cpc!=0:
        Ta2=T-(T-Ta1)*np.exp(-U*A/(mc*Cpc))
        Qr = mc*Cpc*(T-Ta2)*(1-np.exp(-U*A/(mc*Cpc)))
    Qg=0
    deltaH=sum([coeff[i] * Cp[i] * (T - T_ref) for i in range(n_species)])
    dH=[]
    for i in range(n_reactions):
        dH.append(dH_std[i]+deltaH)
    for i in range(n_reactions):
        Qg+=rate_laws(initial_conc, T)[i]*(dH[i])*V
    for i in range(n_species):
            dCdt[i] += r[i]
   
    if reactor_type == "CSTR":
        for i in range(n_species):
            F = [C[i] * flow_rate for i in range(len(C))]
            Qr+=(sum(F))* cp*(T - T0)
            dCdt[i] += (flow_rate / V) * (initial_conc[i] - C[i])
            
        dTdt = (-flow_rate*initial_conc[0] * Cp[i] * (T-T0) +Qg-Qr) / CpS
    elif reactor_type == "Batch":
        dTdt = (Qg-Qr) / CpS
    # else:
    #     N=100
    #     dz=L/N
    #     for i in range(1, N):

    #         dCdt[i] = (-v * (y[i] - y[i-1]))*V/ dz + r
    #         dTdt[i] = (-v * (y[i] - y[i-1]) / dz + (Qg-Qr)) / CpS
    #     # Inlet boundary condition (Dirichlet)
    #         dCdt[0] = 0
    #         dTdt[0] = 0
    T = np.clip(T, 200, 5000)
    return list(dCdt) + [dTdt]

time_span = st.sidebar.number_input("Enter time span: ", value=10.0)
if st.button("Run Simulation"):
    t_span = [0, time_span]
    # T_init = np.full(N, T0)
    # conc_2d = np.array([np.full(N, c) for c in initial_conc])
    # initial_conc=conc_2d.flatten()
    # y_init = initial_conc+T_init  # shape: (n_species+1, n_nodes)
    # y0 = y_init.flatten()
    # # Suppose initial_conc is (n_species, n_nodes), and T0 is scalar
    y0=initial_conc+[T0]
    if reactor_type == "PFR":
        C_init = initial_conc
        T_init = np.full(N, T0)
        y0 = np.concatenate([C_init, T_init])
        sol = solve_ivp(pfr_odes, [0, time_span], y0, method='RK45', dense_output=True)
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
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(t_vals, y_vals[-1], label='T')
        T_vals = y_vals[-1] 
        if mc != 0 and Cpc != 0:
            Ta2 = [T - (T - Ta1) * np.exp(-U * A / (mc * Cpc)) for T in T_vals]
        else:
            Ta2 = T_vals.copy()
        ax1.plot(t_vals, Ta2, label="Ta")
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Temperature (K)")
        ax1.set_title(f"Temperature Profile in {reactor_type}")
        ax1.legend()
        st.pyplot(fig1)  
        final_temp=y_vals[-1] 
        st.write("Ta at t =", Ta2[-1])
        st.write("T(t)= ", final_temp[-1])
    else:
        sol = solve_ivp(odes, [0, time_span], y0, method='RK45', dense_output=True, bounds=(np.zeros_like(y0), np.full_like(y0, np.inf)))

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

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(t_vals, y_vals[-1], label='T')
        T_vals = y_vals[-1] 
        if mc != 0 and Cpc != 0:
            Ta2 = [T - (T - Ta1) * np.exp(-U * A / (mc * Cpc)) for T in T_vals]
        else:
            Ta2 = T_vals.copy()
        ax1.plot(t_vals, Ta2, label="Ta")
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Temperature (K)")
        ax1.set_title(f"Temperature Profile in {reactor_type}")
        ax1.legend()
        final_values = y_vals[:, -1]  # last column

        st.write("Final values at t =", t_vals[-1])
        for i, val in enumerate(final_values[:-1]):
            st.write(f"y[{i}] = {val}")
        st.pyplot(fig1)  
        final_temp=y_vals[-1] 
        st.write("Ta at t =", Ta2[-1])
        st.write("T(t)= ", final_temp[-1])
    
