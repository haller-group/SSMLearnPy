import sympy as sy
import numpy as np

from sympy import latex
from IPython.display import display_latex
from scipy.optimize import minimize


def disp(idx, symObj):
    eqn = '\\[' + idx + ' ' + latex(symObj) + '\\]'
    display_latex(eqn,raw = True)
    return

def dispMore(idx, symObj):
    dispArray = ' '
    for i in range(len(idx)):
        dispArray = dispArray + idx[i] + ' ' + latex(symObj[i])
        
    eqn = '\\[' + dispArray + '\\]'
    display_latex(eqn,raw = True)
    return

def display_equation(coeffs, power, base_symbol='x', complex = False):
    """
    Converts a polynomial expression into symbolic form.  

    Parameters:
    """
    n_variables = power.shape[0]
    variables = [sy.symbols('%s_%d' %(base_symbol, i)) for i in range(n_variables)]
    n_equations = coeffs.shape[0]
    equations = []
    for k in range(n_equations):
        term = 0
        for l,p in enumerate(power.T): 
            prod = 1
            for i, p_ in enumerate(p):
                prod *= variables[i]**p_
            term += coeffs[0,l]*prod
        equations.append(term)
    if complex and n_equations < n_variables: # then we assume that the conjugates are discarded
        n_indep_variables = int(n_variables / 2) # half of them should be conjugates
        variables_conjugate = [sy.symbols('\\bar{%s}_%d' %(base_symbol, i)) for i in range(n_indep_variables)]
        equations = [e.subs([
                (variables[n_indep_variables + i],
                                      variables_conjugate[i])
                                        for i in range(n_indep_variables) ]) 
                                        for e in equations]
        for i in range(len(equations)):
            conj_e = equations[i].conjugate()
            conj_e = conj_e.subs([(variables_conjugate[i].conjugate(), variables[i]) for i in range(n_indep_variables)])
            conj_e = conj_e.subs([(variables[i].conjugate(), variables_conjugate[i]) for i in range(n_indep_variables)])
            equations.append(conj_e)
        return variables[:n_indep_variables], variables_conjugate, equations
    return variables, equations



def convert_to_polar(variables, equations):
    # should have the same number of equations as variables
    n_variables = len(variables)
    n_equations = len(equations)
    ii = sy.sqrt(-1)
    n_polar_variables = int(n_variables / 2)
    radial_variables = [sy.symbols('r_%d' %i) for i in range(n_polar_variables)]
    angle_variables = [sy.symbols('\\varphi_%d' %i) for i in range(n_polar_variables)]

    substituted_equations = []
    
    for e in equations:
        substitution_rules_variable = [(variables[i], radial_variables[i]*sy.exp(ii*angle_variables[i]) ) for i in range(n_polar_variables)]
        substitution_rules_conjugate = [(variables[i+n_polar_variables], radial_variables[i]*sy.exp(-ii*angle_variables[i]) ) for i in range(n_polar_variables)]
        f = e.subs(substitution_rules_variable)
        f = f.subs(substitution_rules_conjugate)
        substituted_equations.append(f)
    # \dot{z} = \dot{r} exp(i \varphi) + i r \dot{\varphi} exp(i \varphi)
    # \dot{\bar{z}} = \dot{r} exp(-i \varphi) - i r \dot{\varphi} exp(-i \varphi)
    # \dot{r} = (\dot{z} exp(-i \varphi) + \dot{\bar{z}} exp(i \varphi)) / 2
    # \dot{\varphi} = (i \dot{z} r exp(-i \varphi) - i \dot{\bar{z}} r exp(i \varphi)) / (2 r^2)
    r_equations = []
    phi_equations = []
    for i in range(n_polar_variables):
        z_dot = substituted_equations[i]
        zbar_dot = substituted_equations[i + n_polar_variables]
        phi_var = angle_variables[i]
        r_var = radial_variables[i]
        r_equations.append(z_dot * sy.exp(-ii * phi_var)/2  +  zbar_dot * sy.exp(ii * phi_var)/2) # dot z * exp(-i \varphi)/2 + dot \bar{z} * exp(i \varphi)/2
        phi_eq_temps = z_dot * sy.exp(-ii * phi_var)/(2*ii) - zbar_dot * sy.exp(ii * phi_var)/(2*ii) # dot z * exp(-i \varphi)/(2i) - dot \bar{z} * exp(i \varphi)/(2i)
        phi_equations.append(phi_eq_temps / (r_var))
    r_equations = [sy.simplify(r) for r in r_equations]
    phi_equations = [sy.simplify(p) for p in phi_equations]

    return radial_variables, angle_variables, r_equations, phi_equations




def backbone_curve_and_damping_curve(r_variables, phidot_eq, rdot_eq):
    # TODO: olny for single DOF for now
    # returns tuples, first element is a callable, second element is the symbolic expression
    backbone_callable = sy.lambdify(r_variables[0], phidot_eq[0])
    damping_callable = sy.lambdify(r_variables[0], -rdot_eq[0]/r_variables[0])
    return (backbone_callable, phidot_eq[0]), (damping_callable, -rdot_eq[0]/r_variables[0])


def extract_FRC(backbone, damping, calibration_amplitude, NonlinearTransform, decoder, observable):
    Npers = 1
    normTimeEval = np.linspace(0,Npers,250*Npers+1)
    normTimeEval = normTimeEval[1:-1]
    phiEval =  normTimeEval*2*np.pi
    cPhi = np.cos(phiEval)
    sPhi = np.sin(phiEval)
    forcingvector = 1j
    zfTemp = forcingvector
    fphase = np.arctan2(np.imag(zfTemp), np.real(zfTemp))




    # Compute roots of the sqrt argument ( (f/r)^2 - a^2(r) )
    backbone_callable, backbone_symbolic = backbone
    damping_callable, damping_symbolic = damping
    rho_var = list(backbone_symbolic.free_symbols)[0]
    rhs_to_eval = lambda r : (backbone_callable(r) *r 
                              - np.sqrt( calibration_amplitude **2 - damping_callable(r)**2 * r**2) 
                              )**2


    roots = sy.solve(calibration_amplitude **2 / rho_var**2 - damping_symbolic**2, rho_var, set=True)
    roots = list(roots[1])
    roots = [r[0].evalf() for r in roots if (r[0].is_real and np.real(r[0].evalf())>0)] # only take the real root
    minimum = minimize(rhs_to_eval, calibration_amplitude, method='Nelder-Mead')
    rho_min_rho_sol = np.array([minimum.x[0], *roots], dtype=np.float64) # roots were sympy objects, convert to float
    rho_out = []
    omega_out = []
    psi_out = []


    u = []
    uPhase = []
    z = []

    nEvalInt = 300
    
    # extract the FRC in normal coordinates
    for i in range(len(rho_min_rho_sol)-1):
        rho_int = np.linspace(rho_min_rho_sol[i], rho_min_rho_sol[i+1], nEvalInt)
        rhoE = rho_int[int(nEvalInt/2)]

        if (calibration_amplitude**2 / rhoE**2 - damping_callable(rhoE)**2) >=0:
            rhoDamp = damping_callable(rho_int)
            rhoFreq = backbone_callable(rho_int)
            rhosqrt = np.array(calibration_amplitude**2/rho_int**2 - damping_callable(rho_int)**2, dtype=float)
            rhoSqrt = np.real(np.sqrt(rhosqrt))
            rhoPhs = np.arctan2(rhoDamp, rhoSqrt).reshape(1,-1)
            rho_out.append(np.repeat(rho_int.reshape(1,-1), 2, axis = 0))
            omega_out.append(np.concatenate(((rhoFreq-rhoSqrt).reshape(1,-1),
                                              (rhoFreq+rhoSqrt).reshape(1,-1))))
            psi_out.append(np.concatenate((rhoPhs+np.pi, -rhoPhs)))
            omega_out = np.squeeze(np.array(omega_out))
            rho_out = np.squeeze(np.array(rho_out))
            psi_out = np.squeeze(np.array(psi_out))
        # transform back to real coordinates:
            for iSol in [1, 0]:
                uTemp = np.zeros((1,nEvalInt))
                uPhaseTemp = np.zeros((1,nEvalInt))
                zTemp = np.zeros((1,nEvalInt))
                for iRho in range(nEvalInt):
                    iOmega = omega_out[-iSol,iRho]
                    timeEval = phiEval/iOmega
                    thetaEval = phiEval + fphase + psi_out[-iSol,iRho]
                    zEval = rho_out[-iSol,iRho]*np.exp(1j*thetaEval)
                    etaEval = NonlinearTransform.inverse_transform(zEval.reshape(-1,1))
                    y = decoder(etaEval)
                    #yAmplitudeFunction = lambda y : y[observable,:]
                    yAmplitudeFunction = y[observable,:]
                    zTemp[iRho] = zEval[0]
                    uTemp[iRho] = np.max(np.abs(yAmplitudeFunction))
                    zfTemp = normTimeEval[1]* np.sum( yAmplitudeFunction*(cPhi-1j*sPhi ) )
                    uPhaseTemp[iRho] = np.arctan2(np.imag(zfTemp), np.real(zfTemp))

                z.append(zTemp)
                u.append(uTemp)
                uPhase.append(uPhaseTemp)

    return rho_out, omega_out, psi_out, z, u, uPhase
    # rhoSol = roots([fliplr(dampcoeffs(1:end-1)),-(fRed(iAmp)*fscale)]);
    # rhoSol = sort(abs(rhoSol(imag(rhoSol)==0))); % eliminate spurios
    # rhoMin = fminsearch(@(r) (freq(r)-sqrt((fRed(iAmp)*fscale./r).^2-damp(r).^2 )).^2,(fRed(iAmp)*fscale)/abs(coeffs(1)));
    # rhoSol = [rhoMin; rhoSol];