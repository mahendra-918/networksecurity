# ⚡ Quick Notes: Electrochemistry (Nernst Equation)

## 1. The Nernst Equation
Used to calculate the **EMF ($E_{cell}$)** of a cell under non-standard conditions (concentration $\neq 1M$, pressure $\neq 1$ bar).

### General Formula (Any Temperature)
$$E_{cell} = E^\circ_{cell} - \frac{2.303 RT}{nF} \log Q$$

### Simplified Formula (At 298 K / 25°C) ⭐ *Most Used*
$$E_{cell} = E^\circ_{cell} - \frac{0.0591}{n} \log Q$$

* **$E^\circ_{cell}$**: Standard Cell Potential ($E^\circ_{cathode} - E^\circ_{anode}$)
* **$n$**: Number of electrons transferred in the balanced equation
* **$Q$**: Reaction Quotient $\left( \frac{[\text{Products}]}{[\text{Reactants}]} \right)$

---

## 2. Writing 'Q' (Reaction Quotient)
For a general reaction:
$$aA(s) + bB(aq) \rightarrow cC(aq) + dD(g)$$

$$Q = \frac{[C]^c (P_D)^d}{[B]^b}$$

> **⚠️ JEE Trap:** pure solids ($s$) and pure liquids ($l$) have active mass = 1. **Ignore them** in the $Q$ expression.

---

## 3. At Equilibrium
When the cell reaches equilibrium:
1.  **$E_{cell} = 0$** (Battery is dead)
2.  **$Q = K_c$** (Equilibrium Constant)
3.  **$\Delta G = 0$**

### Relation between $E^\circ_{cell}$ and $K_c$
$$E^\circ_{cell} = \frac{0.0591}{n} \log K_c$$
*(At 298 K)*

---

## 4. Relation with Gibbs Energy ($\Delta G$)
The maximum work done by the cell:

1.  **Non-Standard:** $\Delta G = -nF E_{cell}$
2.  **Standard:** $\Delta G^\circ = -nF E^\circ_{cell}$

* If $E_{cell} > 0 \Rightarrow \Delta G < 0$ (Spontaneous / Working Cell)
* If $E_{cell} < 0 \Rightarrow \Delta G > 0$ (Non-spontaneous)


## 5. Concentration Cells
A cell made of the **same material** at both electrodes but with **different concentrations**.
* **$E^\circ_{cell} = 0$** (Always)
* For spontaneous working: Concentration at Cathode ($C_2$) > Concentration at Anode ($C_1$).
$$E_{cell} = \frac{0.0591}{n} \log \left( \frac{C_2}{C_1} \right)$$