import sympy as sy
import numpy as np

from sympy import latex
from IPython.display import display_latex



def disp(idx, symObj):
    eqn = '\\[' + idx + ' ' + latex(symObj) + '\\]'
    display_latex(eqn,raw=True)
    return

def dispMore(idx, symObj):
    dispArray = ' '
    for i in range(len(idx)):
        dispArray = dispArray + idx[i] + ' ' + latex(symObj[i])
        
    eqn = '\\[' + dispArray + '\\]'
    display_latex(eqn,raw=True)
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
