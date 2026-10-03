$${\Lambda}_{\mathrm{r}}\left(\mathrm{r}\right)\quad=\quad{\lambda}_{\text{bed}}\left(\mathrm{r}\right)+{\mathrm{K}}_{1}{\mathrm{{Pe}}}_{0}\frac{{\mathrm{u}}_{\mathrm{c}}}{{\bar{\mathrm{u}}}_{0}}\mathrm{f}\left({\mathrm{R}-\mathrm{r}}\right){\lambda}_{\mathrm{f}}\tag{17}$$

The damping function f(R-r) remains the same as for the mass transfer (Eq.(11)), while the slope parameter and the damping parameter are slightly modified to, respectively,

$$K_{1}=\frac{1}{8}\tag{18}$$

$$\mathrm{K}_{2}=0.44+4\exp\left(-\frac{\mathrm{Re}_{0}}{70}\right)\tag{19}$$

The heat release by adsorption, see [19] for $\Delta H_{\mathrm{a d}}$ , is derived in the last term on the right-hand side of Eq. (14) from the change of solids load with time. This very term couples the energy with the mass balance, so that both have to be solved simultaneously in order to account for thermal effects. Heat transfer resistances to or in the particles are neglected. The terms $\delta_{\mathrm{b e d}} ( \mathbf{r} )$ and $\lambda_{\mathrm{b e d}} ( \mathbf{r} )$ in Eqs. (8), (10), (15) and (17) describe the isotropic effective diffusivity and thermal conductivity of the bed without fluid flow. Boundary and initial conditions for Eqs. (7) and (14) are recapitulated in Tab. 1.

On the basis of the above described general model various reductions are possible by neglecting thermal effects, the radial coordinate or gas-to-particle and intraparticle mass transfer resistances. From such reduced versions the following has been considered in more detail in the present work:

1) plug-flow model (1-D) with local equilibrium between the gas and the solids,

2) plug-flow model (1-D) with mass transfer resistance to the solids,

3) 2-D maldistribution model with local equilibrium,

4) 2-D maldistribution model with mass transfer resistance to the solids.

In our terminology “plug flow” means that every influence of the radial coordinate is neglected, including the influence of the wall on porosity and flow velocity. However, axial dispersion, as expressed by the dispersion coefficient $D_{\mathrm{a x}}$ , is accounted for, so that the equation

$$\bar{\psi}\frac{\partial\mathrm{Y}}{\partial\mathrm{t}}=\mathrm{D_{ax}}\frac{\partial^{2}\mathrm{Y}}{\partial\mathrm{z^{2}}}-\bar{u}_{0}\frac{\partial\mathrm{Y}}{\partial\mathrm{z}}-[1-\bar{\psi}]\frac{\partial\mathrm{X}}{\partial\mathrm{t}}\frac{\rho_{\mathrm{p}}}{\rho_{\mathrm{f}}}\tag{20}$$

applies to the isothermal plug flow models (models 1 and 2). Eq. (20) is the classical, conventional way to model packed bed adsorbers. Local equilibrium corresponds, in terms of the two-layers model from [19], to the limiting case of $\beta_{\mathrm{f}} \to\infty$

and $\beta_{\mathrm{p}}\rightarrow\infty.$ At this limit, equilibrium is considered to be sufficient for calculating the response of the solid phase to changes of the concentration in the fluid. Model 4 is our complete, highest order model, as previously outlined and in exact correspondence to [13-18]. Mainly this model has been evaluated for both isothermal and non-isothermal conditions.

# 4 Numerical Solution and its Validation

The partial differential equation or equations of the various models have been solved by the method of lines. The numerical calculations were conducted for different mesh densities, and the results accepted when the change of calculated gas moisture content values was lower than 0.05 % of the maximal difference of gas moisture content appearing in the packed bed. When the error was bigger, the mesh was made denser. Since the width of the concentration front is, in many cases, not much smaller than the length of the bed, equidistant meshes have been used in the axial direction. In the maldistribution models (models 3 and 4 in the previous section) meshes that were denser near the wall than in the center of the tube have been applied.

To check the numerical procedure, respective results have been compared with available analytical solutions. One of such a solution is attributed to Anzelius [1] and refers to model 2 after the classification of section 3, additionally reduced by neglecting axial dispersion $(\mathrm{D}_{\mathrm{a x}}=0).$ . Furthermore, it is assumed that the sorption equilibrium is throughout linear ("Henry's law"), and that the bed is long. The mass transfer resistance is attributed to the fluid phase. Then, axial profiles can be derived to

$$\frac{\mathrm{C}}{\mathrm{C}_{\mathrm{in}}}=\frac{1}{2}\mathrm{erfc}\left(\sqrt{\xi}-\sqrt{\tau}\right)\tag{21}$$

with

$$\xi=6\frac{\beta_{\mathrm{f}}}{\mathrm{d_{p}}}\frac{\mathrm{z}}{\mathrm{u}}\frac{1-\psi}{\psi}\tag{22}$$

and

$$\tau=6\frac{\beta_{\mathrm{f}}}{d_{\mathrm{p}}K}(t-\frac{z}{u})\tag{23}$$

In Eq. (21) the concentration of adsorbate in the gas phase, C, is used instead of the content, Y, assuming an ini-

Table 1. Boundary and initial conditions for models.

<table border="1"><tr><td>t &gt; 0</td><td>$0 \leq r \leq R$</td><td>z = 0</td><td></td><td>\( Y = Y_{\text{in}} \) or \( u_{0}(Y_{\text{in}} - Y) = -D_{\text{ax}} \frac{\partial Y}{\partial z} \)</td><td>T = $T_{\text{in}}$</td></tr><tr><td></td><td></td><td>z = L</td><td>\( \frac{\partial X}{\partial z} = 0 \)</td><td>\( \frac{\partial Y}{\partial z} = 0 \)</td><td>\( \frac{\partial T}{\partial z} = 0 \)</td></tr><tr><td>t &gt; 0</td><td>$0 \leq z \leq L$</td><td>r = 0</td><td>\( \frac{\partial X}{\partial r} = 0 \)</td><td>\( \frac{\partial Y}{\partial r} = 0 \)</td><td>\( \frac{\partial T}{\partial r} = 0 \)</td></tr><tr><td></td><td></td><td>r = R</td><td>\( \frac{\partial X}{\partial r} = 0 \)</td><td>\( \frac{\partial Y}{\partial r} = 0 \)</td><td>T = \( T_{w} \)</td></tr><tr><td>t = 0</td><td>$0 \leq r \leq R$</td><td>$0 \leq z \leq L$</td><td>X(r, z) = X_{0}</td><td>Y(r, z) = Y_{0}</td><td>T(r, z) = T_{0}</td></tr></table>
