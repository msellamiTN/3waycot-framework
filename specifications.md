You will Build the 3WayCoT framework v2.0

The framework will be built on top of the 3WayCoT framework v1.0    
specifications with latex 
\section{Background and Preliminaries}
\subsection{Types of Uncertainty}
Following established statistical terminology \cite{hullermeier2021aleatoric, abdar2021review}, we distinguish between two fundamental types of uncertainty:

\begin{definition}[Aleatoric Uncertainty]
Uncertainty arising from the inherent randomness or noise in the data or the task itself. This form of uncertainty cannot be reduced with additional data or model capacity.
\end{definition}

\begin{definition}[Epistemic Uncertainty]
Uncertainty due to limited knowledge or information about the model or data. This form of uncertainty can potentially be reduced with additional data or improved models.
\end{definition}

In the context of LLMs, we further refine these into:

\begin{itemize}
    \item \textbf{Input Uncertainty}: Ambiguity or incompleteness in the provided prompt or context.
    \item \textbf{Knowledge Uncertainty}: Gaps in the model's learned knowledge or limitations in its ability to access relevant information.
    \item \textbf{Reasoning Uncertainty}: Uncertainty in the logical or inferential steps taken by the model.
    \item \textbf{Output Uncertainty}: Uncertainty in the final response, which may combine all previous forms.
\end{itemize}

\subsection{Evaluation Metrics}
We use the following standard metrics for evaluating uncertainty calibration:

\begin{definition}[Expected Calibration Error (ECE)]
The weighted average of the difference between confidence and accuracy:
\begin{equation}
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
\end{equation}
where $B_m$ represents the $m$-th confidence bin, $n$ is the total number of samples, and $\text{acc}(B_m)$ and $\text{conf}(B_m)$ are the accuracy and average confidence within bin $B_m$, respectively.
\end{definition}

\begin{definition}[Brier Score]
The mean squared error between probabilistic predictions and actual outcomes:
\begin{equation}
\text{BS} = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} (p_{ij} - y_{ij})^2
\end{equation}
where $p_{ij}$ is the predicted probability of instance $i$ belonging to class $j$, $y_{ij}$ is the binary indicator if instance $i$ belongs to class $j$, $N$ is the number of instances, and $C$ is the number of classes.
\end{definition}



\subsection{Triadic Fuzzy Formal Concept Analysis}

We begin by establishing the mathematical foundations of our approach using triadic fuzzy formal concept analysis, which provides a principled framework for representing uncertain reasoning steps. While traditional FCA has been primarily applied to binary relations, our extension to triadic fuzzy contexts enables a more nuanced representation of LLM reasoning processes.

\begin{definition}[Triadic Fuzzy Context]
A triadic fuzzy context is a quadruple $\mathcal{K} = (G, M, B, I)$ where:
\begin{itemize}
    \item $G$ is a set of objects representing reasoning steps
    \item $M$ is a set of attributes representing assumptions underlying reasoning
    \item $B$ is a set of uncertainty conditions
    \item $I: G \times M \times B \rightarrow [0,1]$ is a ternary fuzzy incidence relation
\end{itemize}
The value $I(g, m, b)$ represents the degree to which reasoning step $g$ relies on assumption $m$ under condition $b$.
\end{definition}

\begin{remark}
This triadic structure is particularly well-suited for LLM reasoning as it captures not only the relationship between reasoning steps and their underlying assumptions, but also how these relationships vary under different conditions or contexts. The fuzzy incidence relation allows for degrees of reliance rather than binary dependencies.
\end{remark}
 

Building on this definition, we construct triadic concepts that capture relationships between reasoning steps, logical properties, and uncertainty conditions through derivation operators:
\subsection{Triadic Derivation Operators}

Building on this definition, we construct triadic concepts that capture relationships between reasoning steps, underlying assumptions, and uncertainty conditions through derivation operators:

\begin{definition}[Triadic Derivation Operators]
For a triadic fuzzy context $(G, M, B, I)$ and a threshold $\tau \in [0,1]$, we define the following derivation operators:

For $X \subseteq G$, $Y \subseteq M$, $Z \subseteq B$:
\begin{align}
X^{(1)}_{YZ} &= \{m \in M \mid \forall g \in X, \forall b \in Z: I(g, m, b) \geq \tau\} \\
Y^{(2)}_{XZ} &= \{b \in B \mid \forall g \in X, \forall m \in Y: I(g, m, b) \geq \tau\} \\
Z^{(3)}_{XY} &= \{g \in G \mid \forall m \in Y, \forall b \in Z: I(g, m, b) \geq \tau\}
\end{align}
\end{definition}

These operators now have clearer interpretations:

\begin{itemize}
    \item $X^{(1)}_{YZ}$ identifies assumptions that are consistently relied upon by a set of reasoning steps under specific conditions
    \item $Y^{(2)}_{XZ}$ identifies conditions under which certain assumptions are consistently relied upon by reasoning steps
    \item $Z^{(3)}_{XY}$ identifies reasoning steps that consistently rely on certain assumptions under specific conditions
\end{itemize}

\subsection{Example: Medical Diagnosis Reasoning}

\begin{example}[Medical Diagnosis Reasoning]
Consider an LLM reasoning about a medical diagnosis with:
\begin{itemize}
    \item $G = \{S_1: \text{"Patient has fever"}, S_2: \text{"Patient likely has infection"}, S_3: \text{"Recommend antibiotics"}\}$
    \item $M = \{m_1: \text{"fever indicates infection"}, m_2: \text{"infections are bacterial"}, m_3: \text{"antibiotics are appropriate first-line treatment"}\}$
    \item $B = \{b_1: \text{"general medicine"}, b_2: \text{"pediatrics"}\}$
\end{itemize}

The fuzzy incidence relation might assign $I(S_2, m_1, b_1) = 0.8$ indicating that the reasoning step "Patient likely has infection" relies strongly on the assumption that "fever indicates infection" in general medicine contexts, but perhaps $I(S_2, m_1, b_2) = 0.6$ in pediatric contexts where fever might have different implications.

For a threshold $\tau = 0.7$, the derivation $\{S_2\}^{(1)}_{\{b_1\}} = \{m_1, m_2\}$ would indicate that in general medicine contexts, the step about infection relies strongly on both the assumptions that fever indicates infection and that infections are bacterial.
\end{example}

 
\begin{definition}[Triadic Concept]
A triadic concept of a triadic fuzzy context $(G, M, B, I)$ with threshold $\tau$ is a triple $(A, C, D)$ with $A \subseteq G$, $C \subseteq M$, and $D \subseteq B$ that satisfies:
\begin{align}
A &= Z^{(3)}_{CD} \\
C &= X^{(1)}_{AD} \\
D &= Y^{(2)}_{AC}
\end{align}
\end{definition}

Each triadic concept now represents a maximal group of reasoning steps that consistently rely on a specific set of assumptions under certain conditions. This provides a formal structure for analyzing the assumption-based dependencies in Chain-of-Thought reasoning.

The set of all triadic concepts forms a structure with important mathematical properties:

\begin{theorem}[Completeness of Triadic Concept Lattice]
The set of all triadic concepts of a triadic fuzzy context $(G, M, B, I)$ with threshold $\tau$, denoted by $\mathfrak{B}(G, M, B, I, \tau)$, forms a complete lattice when ordered by:
\begin{align}
(A_1, C_1, D_1) \leq (A_2, C_2, D_2) \iff A_1 \subseteq A_2 \iff C_1 \supseteq C_2 \land D_1 \supseteq D_2
\end{align}
\end{theorem}

\begin{proof}
We prove this by showing that for any set of triadic concepts, there exists a unique supremum and infimum in $\mathfrak{B}(G, M, B, I, \tau)$.

Let $\{(A_j, C_j, D_j) \mid j \in J\}$ be a set of triadic concepts. We define:
\begin{align}
A' &= \bigcup_{j \in J} A_j \\
C' &= \bigcap_{j \in J} C_j \\
D' &= \bigcap_{j \in J} D_j
\end{align}

Then $(A', C', D')$ is not necessarily a triadic concept. However, we can define:
\begin{align}
A^* &= Z^{(3)}_{C'D'} \\
C^* &= X^{(1)}_{A^*D'} \\
D^* &= Y^{(2)}_{A^*C^*}
\end{align}

The triple $(A^*, C^*, D^*)$ is the supremum of the original set of concepts in the lattice.

The construction of the infimum follows similarly by starting with the intersection of objects and unions of attributes and conditions.

Since every subset of concepts has both a supremum and an infimum, $\mathfrak{B}(G, M, B, I, \tau)$ forms a complete lattice.
\end{proof}

\begin{remark}
The completeness property is crucial as it ensures that the concept space is well-structured and supports operations like concept generalization and specialization in a consistent manner. This provides theoretical guarantees for the stability of our reasoning framework.
\end{remark}

\subsection{Interval-Valued Three-Way Decisions}

To properly model epistemic uncertainty in the reasoning process, we extend the traditional three-way decision framework \cite{yao2010three} to incorporate interval-valued fuzzy sets. This approach draws on theoretical work in the field of imprecise probabilities and decision making under uncertainty \cite{troffaes2014lower}.

\begin{definition}[Interval-Valued Fuzzy Set]
An interval-valued fuzzy set $A$ on universe $X$ is defined as:
\begin{align}
A = \{(x, [\mu^L_A(x), \mu^U_A(x)]) \mid x \in X\}
\end{align}
where $\mu^L_A(x)$ and $\mu^U_A(x)$ represent the lower and upper bounds, respectively, of the membership degree of $x$ in $A$, satisfying $0 \leq \mu^L_A(x) \leq \mu^U_A(x) \leq 1$.
\end{definition}

\begin{remark}
Unlike traditional fuzzy sets that assign a single membership degree, interval-valued fuzzy sets capture a range of possible membership values. This is particularly appropriate for LLM reasoning, where the model may have varying degrees of certainty about different reasoning steps.
\end{remark}

For each reasoning step $S_i$, we define two interval-valued fuzzy sets:

\begin{align}
\text{Accept}(S_i) &= \{(S_i, [\mu^L_{\text{accept}}(S_i), \mu^U_{\text{accept}}(S_i)])\} \\
\text{Reject}(S_i) &= \{(S_i, [\mu^L_{\text{reject}}(S_i), \mu^U_{\text{reject}}(S_i)])\}
\end{align}

\begin{example}[Continued from Medical Example]
For the reasoning step $S_2$: "Patient likely has infection", we might have:
\begin{align}
\text{Accept}(S_2) &= \{(S_2, [0.7, 0.9])\} \\
\text{Reject}(S_2) &= \{(S_2, [0.1, 0.3])\}
\end{align}

This indicates high confidence in accepting this reasoning step, with a relatively narrow uncertainty interval $[0.7, 0.9]$, and low confidence in rejecting it, also with a narrow interval $[0.1, 0.3]$. In contrast, for a more uncertain step $S_3$: "Recommend antibiotics", we might have:
\begin{align}
\text{Accept}(S_3) &= \{(S_3, [0.4, 0.8])\} \\
\text{Reject}(S_3) &= \{(S_3, [0.2, 0.6])\}
\end{align}

The wider intervals reflect greater uncertainty about whether to accept or reject this recommendation.
\end{example}

\begin{definition}[Three-Way Decision Rule]
For thresholds $\alpha, \beta \in [0,1]$, the three-way decision rule for reasoning step $S_i$ is defined as:
\begin{align}
d(S_i) = 
\begin{cases}
\text{accept} & \text{if } \mu^L_{\text{accept}}(S_i) \geq \alpha \\
\text{reject} & \text{if } \mu^L_{\text{reject}}(S_i) \geq \beta \\
\text{abstain} & \text{otherwise}
\end{cases}
\end{align}
\end{definition}

\begin{proposition}[Decision Region Separation]
If $\alpha + \beta > 1$, then the acceptance and rejection regions are disjoint, i.e., no reasoning step can be simultaneously accepted and rejected.
\end{proposition}

\begin{proof}
Suppose that a reasoning step $S_i$ is both accepted and rejected. Then:
\begin{align}
\mu^L_{\text{accept}}(S_i) \geq \alpha \text{ and } \mu^L_{\text{reject}}(S_i) \geq \beta
\end{align}

Since membership functions must satisfy $\mu_{\text{accept}}(S_i) + \mu_{\text{reject}}(S_i) \leq 1$ for any point in the universe, we have:
\begin{align}
\mu^L_{\text{accept}}(S_i) + \mu^L_{\text{reject}}(S_i) \leq 1
\end{align}

Therefore:
\begin{align}
\alpha + \beta \leq \mu^L_{\text{accept}}(S_i) + \mu^L_{\text{reject}}(S_i) \leq 1
\end{align}

This contradicts our assumption that $\alpha + \beta > 1$. Hence, no reasoning step can be simultaneously accepted and rejected.
\end{proof}

\begin{remark}
This proposition establishes an important theoretical guarantee: with appropriate threshold settings, our framework will never produce contradictory decisions about reasoning steps. The abstention region serves as a buffer between acceptance and rejection, allowing the system to explicitly acknowledge uncertainty rather than making potentially erroneous binary decisions.
\end{remark}

\subsection{Integration of Triadic FCA with Three-Way Decisions}

We now establish the connection between triadic concepts and interval-valued three-way decisions. This integration forms the core of our 3WayCOT framework.

\begin{definition}[Projection Function]
Let $(A, C, D)$ be a triadic concept. The projection function $\pi$ maps this concept to acceptance and rejection intervals as follows:
\begin{align}
\pi(A, C, D) = ([\mu^L_{\text{accept}}, \mu^U_{\text{accept}}], [\mu^L_{\text{reject}}, \mu^U_{\text{reject}}])
\end{align}
where:
\begin{align}
\mu^L_{\text{accept}} &= \min_{g \in A, m \in C_{\text{accept}}, b \in D} I(g, m, b) \\
\mu^U_{\text{accept}} &= \max_{g \in A, m \in C_{\text{accept}}, b \in D} I(g, m, b) \\
\mu^L_{\text{reject}} &= \min_{g \in A, m \in C_{\text{reject}}, b \in D} I(g, m, b) \\
\mu^U_{\text{reject}} &= \max_{g \in A, m \in C_{\text{reject}}, b \in D} I(g, m, b)
\end{align}
with $C_{\text{accept}}$ and $C_{\text{reject}}$ being the subsets of attributes $M$ that correspond to acceptance and rejection properties, respectively.
\end{definition}

\begin{example}[Complete Running Example]
We now fully develop our medical diagnosis example to illustrate the complete framework:

 
Let's consider a simple triadic context with:
\begin{itemize}
    \item $G = \{S_1, S_2, S_3\}$ where:
        \begin{itemize}
            \item $S_1$: "Patient has fever of 102°F"
            \item $S_2$: "Patient likely has bacterial infection"
            \item $S_3$: "Recommend antibiotics treatment"
        \end{itemize}
    \item $M = \{m_1, m_2, m_3\}$ where:
        \begin{itemize}
            \item $m_1$: "high fever suggests bacterial infection" (in $C_{\text{abstain}}$)
            \item $m_2$: "bacterial infections require antibiotics" (in $C_{\text{accept}}$)
            \item $m_3$: "antibiotics have dangerous side effects" (in $C_{\text{reject}}$)
        \end{itemize}
    \item $B = \{b_1, b_2\}$ where:
        \begin{itemize}
            \item $b_1$: "general medicine context"
            \item $b_2$: "patient has penicillin allergy"
        \end{itemize}
\end{itemize}


With incidence relation values (simplified):

For context $b_1$ (general medicine):
\begin{align}
I(S_1, m_1, b_1) &= 0.9 \quad I(S_1, m_2, b_1) = 0.8 \quad I(S_1, m_3, b_1) = 0.1 \\
I(S_2, m_1, b_1) &= 0.7 \quad I(S_2, m_2, b_1) = 0.8 \quad I(S_2, m_3, b_1) = 0.2 \\
I(S_3, m_1, b_1) &= 0.7 \quad I(S_3, m_2, b_1) = 0.8 \quad I(S_3, m_3, b_1) = 0.3
\end{align}

For context $b_2$ (penicillin allergy):
\begin{align}
I(S_1, m_1, b_2) &= 0.9 \quad I(S_1, m_2, b_2) = 0.8 \quad I(S_1, m_3, b_2) = 0.1 \\
I(S_2, m_1, b_2) &= 0.7 \quad I(S_2, m_2, b_2) = 0.7 \quad I(S_2, m_3, b_2) = 0.2 \\
I(S_3, m_1, b_2) &= 0.5 \quad I(S_3, m_2, b_2) = 0.6 \quad I(S_3, m_3, b_2) = 0.8
\end{align}

For a specific triadic concept $(A, C, D) = (\{S_1, S_2\}, \{m_1, m_2\}, \{b_1\})$, the projection function gives:
\begin{align}
\mu^L_{\text{accept}} &= \min(0.9, 0.8, 0.7, 0.8) = 0.7 \\
\mu^U_{\text{accept}} &= \max(0.9, 0.8, 0.7, 0.8) = 0.9 \\
\mu^L_{\text{reject}} &= \min(0.1, 0.2) = 0.1 \\
\mu^U_{\text{reject}} &= \max(0.1, 0.2) = 0.2
\end{align}

With acceptance threshold $\alpha = 0.6$ and rejection threshold $\beta = 0.6$, the decision would be to accept this concept, as $\mu^L_{\text{accept}} = 0.7 > \alpha$ and $\mu^L_{\text{reject}} = 0.1 < \beta$.

For another concept $(A', C', D') = (\{S_3\}, \{m_1, m_2\}, \{b_2\})$ (antibiotics recommendation for patient with penicillin allergy):
\begin{align}
\mu^L_{\text{accept}} &= \min(0.5, 0.6) = 0.5 \\
\mu^U_{\text{accept}} &= \max(0.5, 0.6) = 0.6 \\
\mu^L_{\text{reject}} &= 0.8 \\
\mu^U_{\text{reject}} &= 0.8
\end{align}

This would lead to rejection as $\mu^L_{\text{reject}} = 0.8 > \beta$ and $\mu^L_{\text{accept}} = 0.5 < \alpha$, correctly identifying that antibiotics recommendation is potentially harmful in the context of a penicillin allergy.
\end{example}

The uncertainty measure for each reasoning step $S_i$ is defined as:
\begin{align}
\text{Uncertainty}(S_i) = (\mu^U_{\text{accept}}(S_i) - \mu^L_{\text{accept}}(S_i)) + (\mu^U_{\text{reject}}(S_i) - \mu^L_{\text{reject}}(S_i))
\end{align}

\begin{theorem}[Monotonicity of Uncertainty]
Let $(A_1, C_1, D_1) \leq (A_2, C_2, D_2)$ be two triadic concepts. If $A_1 \subset A_2$, then $\text{Uncertainty}(S_i^1) \leq \text{Uncertainty}(S_i^2)$ for any reasoning step $S_i \in A_1 \cap A_2$, where $S_i^j$ denotes the reasoning step $S_i$ in the context of concept $(A_j, C_j, D_j)$.
\end{theorem}

\begin{proof}
Since $(A_1, C_1, D_1) \leq (A_2, C_2, D_2)$, we have $A_1 \subset A_2$, $C_1 \supset C_2$, and $D_1 \supset D_2$.

For any reasoning step $S_i \in A_1 \cap A_2$, the set of attribute-condition pairs considered in concept $(A_1, C_1, D_1)$ is larger than in $(A_2, C_2, D_2)$ because $C_1 \times D_1 \supset C_2 \times D_2$.

A larger set of attribute-condition pairs leads to potentially smaller minimum values and larger maximum values in the computation of membership intervals. Therefore:
\begin{align}
\mu^L_{\text{accept}}(S_i^1) &\leq \mu^L_{\text{accept}}(S_i^2) \\
\mu^U_{\text{accept}}(S_i^1) &\geq \mu^U_{\text{accept}}(S_i^2) \\
\mu^L_{\text{reject}}(S_i^1) &\leq \mu^L_{\text{reject}}(S_i^2) \\
\mu^U_{\text{reject}}(S_i^1) &\geq \mu^U_{\text{reject}}(S_i^2)
\end{align}

This implies:
\begin{align}
\mu^U_{\text{accept}}(S_i^1) - \mu^L_{\text{accept}}(S_i^1) &\geq \mu^U_{\text{accept}}(S_i^2) - \mu^L_{\text{accept}}(S_i^2) \\
\mu^U_{\text{reject}}(S_i^1) - \mu^L_{\text{reject}}(S_i^1) &\geq \mu^U_{\text{reject}}(S_i^2) - \mu^L_{\text{reject}}(S_i^2)
\end{align}

Therefore:
\begin{align}
\text{Uncertainty}(S_i^1) \geq \text{Uncertainty}(S_i^2)
\end{align}

This completes the proof.
\end{proof}
\begin{figure}[!h]
\centering
\begin{tikzpicture}[
    scale=0.85,
    concept/.style={
        rectangle, 
        draw, 
        rounded corners, 
        minimum height=1.2cm, 
        minimum width=2.8cm,
        text centered,
        font=\footnotesize,
        align=center
    },
    decision/.style={font=\scriptsize\bfseries},
    arrow/.style={->, thick, >=stealth}
]

% Color-coded nodes
\node[concept, fill=blue!20] (c1) at (0,6) {$(\{g_1\}, \{m_1,m_2\}, \{b_1,b_2\})$};
\node[concept, fill=orange!20] (c2) at (-4,3) {$(\{g_1,g_2\}, \{m_1\}, \{b_1,b_2\})$};
\node[concept, fill=blue!20] (c3) at (0,3.5) {$(\{g_1\}, \{m_1,m_2\}, \{b_1\})$};
\node[concept, fill=orange!20] (c4) at (4,3) {$(\{g_1,g_3\}, \{m_1,m_2\}, \{b_2\})$};
\node[concept, fill=red!20] (c5) at (-4,0.5) {$(\{g_1,g_2,g_3\}, \{m_1\}, \{b_1\})$};
\node[concept, fill=orange!20] (c6) at (0,1) {$(\{g_1,g_2\}, \{m_1\}, \{b_1\})$};
\node[concept, fill=red!20] (c7) at (4,0.5) {$(\{g_1,g_3\}, \{m_1\}, \{b_2\})$};
\node[concept, fill=orange!20] (c8) at (0,-2.5) {$(\{g_1,g_2,g_3\}, \emptyset, \{b_1,b_2\})$};

% Non-overlapping curved connections
\draw[arrow] (c1) to[out=190,in=90] (c2);
\draw[arrow] (c1) to[out=-90,in=90] (c3);
\draw[arrow] (c1) to[out=-10,in=90] (c4);
\draw[arrow] (c2) to[out=-80,in=150] (c5);
\draw[arrow] (c2) to[out=-30,in=180] (c6);
\draw[arrow] (c3) to[out=-80,in=90] (c5);
\draw[arrow] (c3) to[out=-60,in=90] (c6);
\draw[arrow] (c4) to[out=-80,in=90] (c7);
\draw[arrow] (c5) to[out=-30,in=190] (c8);
\draw[arrow] (c6) to[out=-90,in=90] (c8);
\draw[arrow] (c7) to[out=-90,in=10] (c8);

% Color-coded decision labels with adjusted positions
\node[decision, blue, right=0.2cm of c1] {Accept: [0.8, 0.9]};
\node[decision, blue, below=0.2cm of c3] {Accept: [0.7, 0.8]};
\node[decision, orange, left=0.2cm of c2] {Abstain: [0.4, 0.6]};
\node[decision, orange, right=0.2cm of c4] {Abstain: [0.5, 0.7]};
\node[decision, red, left=0.2cm of c5] {Reject: [0.7, 0.8]};
\node[decision, red, right=0.2cm of c7] {Reject: [0.8, 0.9]};
\node[decision, orange, below=0.2cm of c6] {Abstain: [0.3, 0.5]};
\node[decision, orange, below=0.2cm of c8] {Abstain: [0.2, 0.4]};

% Title and legend
\node[font=\large, above=0.5cm of c1] {Triadic Concept Lattice with Three-Way Decisions};
\node[draw, fill=white, rounded corners, anchor=north] at (c8.south) [yshift=-1cm] {
    \scriptsize
    \textcolor{blue}{Accept} \quad 
    \textcolor{orange}{Abstain} \quad 
    \textcolor{red}{Reject} \quad 
    Thresholds: $\alpha = \beta = 0.7$
};

\end{tikzpicture}
\caption{Triadic concept lattice visualization with color-coded decision regions and non-overlapping layout. Concepts are categorized as Accept (blue), Abstain (orange), and Reject (red) based on threshold parameters $\alpha = \beta = 0.7$.}
\label{fig:concept_lattice}
\end{figure}
\subsection{Chain-of-Thought Reasoning}

CoT decomposes complex reasoning into explicit intermediate steps \cite{wei2022chain}:

$$y = f_\text{CoT}(x) = f_\text{answer}(f_\text{reason}(x))$$

where $x$ is the input, $f_\text{reason}$ generates reasoning steps $S = \{S_1, S_2, ..., S_n\}$, and $f_\text{answer}$ produces the final answer based on these steps.
\subsection{System Architecture}

The 3WayCOT system consists of four main components designed to work in concert to provide uncertainty-aware reasoning:

\begin{enumerate}
    \item \textbf{Chain-of-Thought Generator}: Produces initial reasoning steps $S = \{S_1, S_2, \ldots, S_n\}$ for a given query.
    \item \textbf{Triadic Context Constructor}: Builds a triadic fuzzy context for each reasoning step and then generate Fuzzy Triadic Lattice.
    \item \textbf{Three-Way Decision Maker}: Assigns decisions (accept, reject, abstain) to each step.
    \item \textbf{Uncertainty Resolution Module}: Refines steps with high uncertainty through retrieval or backtracking.
\end{enumerate}

\begin{figure}[!t]
\centering
\begin{tikzpicture}[
    module/.style={rectangle, draw=black!80, rounded corners, 
                  minimum width=5cm, minimum height=1.2cm, 
                  text centered, font=\small, fill=blue!10},
    io/.style={trapezium, trapezium left angle=70, trapezium right angle=110, 
              draw=blue!80, minimum width=5cm, minimum height=1.5cm, 
              text centered, font=\small, fill=blue!20},
    arrow/.style={thick, ->, >=stealth},
    database/.style={cylinder, draw=green!80, shape border rotate=90, 
                    aspect=0.3, minimum height=1cm, minimum width=1.5cm, 
                    text centered, font=\small, fill=green!20},
    decision/.style={module, fill=orange!20},
    resolution/.style={module, fill=purple!20},
    cot/.style={module, fill=red!20}
]

% Color-coded nodes
\node[io] (input) at (0,0) {Query $Q$, Context $C$};
\node[io] (output) at (6,-9) {Answer $y$ with uncertainty bounds};

\node[cot] (cot) at (0,-2) {Chain-of-Thought Generator};
\node[module] (context) at (0,-4.5) {Triadic Context Constructor};
\node[decision] (decision) at (0,-7) {Three-Way Decision Maker};
\node[resolution] (resolution) at (5,-5) {Uncertainty Resolution Module};

\node[database] (knowledge) at (-3,-3) {Knowledge Base};
\node[database] (lattice) at (5,-2.5) {Concept Lattice $\mathcal{L}$};

% Color-coded connections
\draw[arrow, blue!50] (input) -- (cot);
\draw[arrow, red!50] (cot) -- node[right, font=\tiny] {Steps $S_1,\ldots,S_n$} (context);
\draw[arrow, blue!50] (context) -- node[right, font=\tiny] {$(G_i,M_i,B_i,I_i)$} (decision);
\draw[arrow, orange!50] (decision) -- node[below right, font=\tiny] {Accept/Reject/Abstain} (output);
\draw[arrow, green!50] (knowledge) -- (context.north west);
\draw[arrow, blue!50] (context) -- (lattice);
\draw[arrow, green!50] (lattice) -- (decision.north east);
\draw[arrow, orange!50] (decision.east) -- node[below, font=\tiny] {Abstain} (resolution.south west);
\draw[arrow, purple!50] (resolution) -- (context.east);
\draw[arrow, green!50] (knowledge.east) -- (resolution.west);

% Colored feedback loop
\draw[arrow, purple!50] (resolution) to[out=170,in=10] node[below left, font=\tiny] {Backtrack} (cot);

% System boundary
\draw[dashed, rounded corners, gray] (-4,-1) rectangle (7,-8.5);
\node[anchor=north west, gray] at (-4,-1) {3WayCOT System};

\subsection{Chain-of-Thought Reasoning}

CoT decomposes complex reasoning into explicit intermediate steps \cite{wei2022chain}:

$$y = f_\text{CoT}(x) = f_\text{answer}(f_\text{reason}(x))$$

where $x$ is the input, $f_\text{reason}$ generates reasoning steps $S = \{S_1, S_2, ..., S_n\}$, and $f_\text{answer}$ produces the final answer based on these steps.
\subsection{System Architecture}

The 3WayCOT system consists of four main components designed to work in concert to provide uncertainty-aware reasoning:

\begin{enumerate}
    \item \textbf{Chain-of-Thought Generator}: Produces initial reasoning steps $S = \{S_1, S_2, \ldots, S_n\}$ for a given query.
    \item \textbf{Triadic Context Constructor}: Builds a triadic fuzzy context for each reasoning step.
    \item \textbf{Three-Way Decision Maker}: Assigns decisions (accept, reject, abstain) to each step.
    \item \textbf{Uncertainty Resolution Module}: Refines steps with high uncertainty through retrieval or backtracking.
\end{enumerate}

\begin{figure}[!t]
\centering
\begin{tikzpicture}[
    module/.style={rectangle, draw=black!80, rounded corners, 
                  minimum width=5cm, minimum height=1.2cm, 
                  text centered, font=\small, fill=blue!10},
    io/.style={trapezium, trapezium left angle=70, trapezium right angle=110, 
              draw=blue!80, minimum width=5cm, minimum height=1.5cm, 
              text centered, font=\small, fill=blue!20},
    arrow/.style={thick, ->, >=stealth},
    database/.style={cylinder, draw=green!80, shape border rotate=90, 
                    aspect=0.3, minimum height=1cm, minimum width=1.5cm, 
                    text centered, font=\small, fill=green!20},
    decision/.style={module, fill=orange!20},
    resolution/.style={module, fill=purple!20},
    cot/.style={module, fill=red!20}
]

% Color-coded nodes
\node[io] (input) at (0,0) {Query $Q$, Context $C$};
\node[io] (output) at (6,-9) {Answer $y$ with uncertainty bounds};

\node[cot] (cot) at (0,-2) {Chain-of-Thought Generator};
\node[module] (context) at (0,-4.5) {Triadic Context Constructor};
\node[decision] (decision) at (0,-7) {Three-Way Decision Maker};
\node[resolution] (resolution) at (5,-5) {Uncertainty Resolution Module};

\node[database] (knowledge) at (-3,-3) {Knowledge Base};
\node[database] (lattice) at (5,-2.5) {Concept Lattice $\mathcal{L}$};

% Color-coded connections
\draw[arrow, blue!50] (input) -- (cot);
\draw[arrow, red!50] (cot) -- node[right, font=\tiny] {Steps $S_1,\ldots,S_n$} (context);
\draw[arrow, blue!50] (context) -- node[right, font=\tiny] {$(G_i,M_i,B_i,I_i)$} (decision);
\draw[arrow, orange!50] (decision) -- node[below right, font=\tiny] {Accept/Reject/Abstain} (output);
\draw[arrow, green!50] (knowledge) -- (context.north west);
\draw[arrow, blue!50] (context) -- (lattice);
\draw[arrow, green!50] (lattice) -- (decision.north east);
\draw[arrow, orange!50] (decision.east) -- node[below, font=\tiny] {Abstain} (resolution.south west);
\draw[arrow, purple!50] (resolution) -- (context.east);
\draw[arrow, green!50] (knowledge.east) -- (resolution.west);

% Colored feedback loop
\draw[arrow, purple!50] (resolution) to[out=170,in=10] node[below left, font=\tiny] {Backtrack} (cot);

% System boundary
\draw[dashed, rounded corners, gray] (-4,-1) rectangle (7,-8.5);
\node[anchor=north west, gray] at (-4,-1) {3WayCOT System};

\end{tikzpicture}
\caption{System architecture of 3WayCOT showing the interaction between components. The color coding highlights different functional areas: blue for data flow, red for reasoning generation, orange for decision making, purple for uncertainty resolution, and green for knowledge integration.}
\label{fig:system_architecture_color}
\end{figure}

Figure \ref{fig:system_architecture_color} illustrates the architecture and data flow within the 3WayCOT system. Each component plays a critical role in transforming an initial query into an uncertainty-aware answer:

\begin{itemize}
    \item The \textbf{Chain-of-Thought Generator} leverages an LLM to produce a sequence of reasoning steps $S_1, S_2, \ldots, S_n$ that lead from the query $Q$ to a potential answer $y$. Unlike standard CoT approaches, our generator is prompted to articulate assumptions and highlight potential areas of uncertainty.
    
    \item The \textbf{Triadic Context Constructor} transforms each reasoning step into a formal representation within a triadic fuzzy context and build triadic lattice form the context. This involves mapping the step's attributes to formal properties and establishing membership degrees based on knowledge base lookups and model confidence scores.
    
    \item The \textbf{Three-Way Decision Maker} evaluates each reasoning step using interval-valued fuzzy sets and applies the three-way decision rule to classify the step as accepted, rejected, or abstained.
    
    \item The \textbf{Uncertainty Resolution Module} is triggered when reasoning steps are assigned to the abstention region. It employs two main strategies to resolve uncertainty: retrieval from external knowledge sources and backtracking to revise earlier reasoning steps.
\end{itemize}

A key feature of our architecture is the feedback loop that allows for iterative refinement of reasoning steps when uncertainty is high. This design enables the system to adapt its reasoning process based on uncertainty assessments, rather than forcing premature conclusions.

\subsection{3WayCot Algorithms}

We now present the detailed algorithms for each component of the 3WayCOT framework. Algorithm \ref{alg:main} provides the high-level orchestration of the system, while subsequent algorithms detail the specific operations of each component.

\begin{algorithm}[!ht]
\caption{3WayCOT: Main Algorithm}
\label{alg:main}
\begin{algorithmic}[1]
\Require Query $Q$, context $C$, thresholds $\alpha, \beta \in [0,1]$, threshold $\tau \in [0,1]$
\Ensure Answer $y$ with uncertainty quantification

\State $S \leftarrow \text{ReasoningGenerator}(Q, C)$ \Comment{Generate initial reasoning steps}
\State $\mathcal{K} \leftarrow \text{TriadicContextBuilder}(S, \text{KB})$ \Comment{Build triadic context from knowledge base}
\State $\mathcal{L} \leftarrow \text{ConceptLatticeBuilder}(\mathcal{K}, \tau)$ \Comment{Construct concept lattice with threshold}
\State $U \leftarrow \emptyset$ \Comment{Initialize set of uncertain steps}

\For{each reasoning step $S_i \in S$}
    \State $[\mu^L_{\text{accept}}, \mu^U_{\text{accept}}], [\mu^L_{\text{reject}}, \mu^U_{\text{reject}}] \leftarrow \text{MembershipCalculator}(S_i, \mathcal{L})$
    \State $d_i \leftarrow \text{ThreeWayDecision}([\mu^L_{\text{accept}}, \mu^U_{\text{accept}}], [\mu^L_{\text{reject}}, \mu^U_{\text{reject}}], \alpha, \beta)$
    \If{$d_i = \text{abstain}$}
        \State $U \leftarrow U \cup \{S_i\}$ \Comment{Add to uncertain steps}
    \EndIf
\EndFor

\If{$U \neq \emptyset$}
    \State $S' \leftarrow \text{UncertaintyResolver}(U, S, \mathcal{K}, \text{KB})$ \Comment{Resolve uncertain steps}
    \If{$S' \neq S$}
        \State \Return{$\text{3WayCOT}(Q, C, \alpha, \beta, \tau)$} \Comment{Recursively update with new steps}
    \EndIf
\EndIf

\State $y \leftarrow \text{Aggregator}(S, \{d_i\}, \{[\mu^L_{\text{accept}}, \mu^U_{\text{accept}}]\}, \{[\mu^L_{\text{reject}}, \mu^U_{\text{reject}}]\})$
\State \Return{$y$ with uncertainty bounds}
\end{algorithmic}
\end{algorithm}

\subsubsection{Chain-of-Thought Generator}

The Chain-of-Thought Generator produces a sequence of reasoning steps that connect the query to a potential answer. Algorithm \ref{alg:reasoning_generator} details this process.

\begin{algorithm}[H]
\caption{ReasoningGenerator}
\label{alg:reasoning_generator}
\begin{algorithmic}[1]
\Require Query $Q$, context $C$
\Ensure Sequence of reasoning steps $S = \{S_1, S_2, \ldots, S_n\}$

\State $P \leftarrow \text{ConstructPrompt}(Q, C)$ \Comment{Create uncertainty-aware CoT prompt}
\State $R \leftarrow \text{QueryLLM}(P)$ \Comment{Get raw response from LLM}
\State $S \leftarrow \text{ExtractReasoningSteps}(R)$ \Comment{Parse response into discrete steps}

\For{each step $S_i \in S$}
    \State $A_i \leftarrow \text{ExtractAssumptions}(S_i)$ \Comment{Identify implicit assumptions}
    \State $S_i \leftarrow \text{AnnotateWithAssumptions}(S_i, A_i)$ \Comment{Annotate step with assumptions}
\EndFor

\State \Return{$S$}
\end{algorithmic}
\end{algorithm}

The key innovation in our Chain-of-Thought Generator is the explicit annotation of assumptions within each reasoning step. By identifying and making these assumptions explicit, we create a foundation for more accurate uncertainty quantification.

\begin{example}[Assumption Annotation in Medical Reasoning]
For a query about antibiotic treatment for a patient with reported fever:

\begin{itemize}
    \item Initial step $S_2$: "Patient likely has bacterial infection"
    \item Annotated step $S_2$: "Patient likely has bacterial infection 
    \item [Assumptions: fever is bacterial rather than viral in origin; no recent viral exposure reported; patient has no history of autoimmune disorders that could cause fever]"
\end{itemize}

These explicit assumptions become crucial inputs for the Triadic Context Constructor.
\end{example}

\subsubsection{Triadic Context Constructor}

The Triadic Context Constructor transforms reasoning steps into a formal triadic fuzzy context. Algorithm \ref{alg:context_builder} outlines this process.

\begin{algorithm}[H]
\caption{TriadicContextBuilder}
\label{alg:context_builder}
\begin{algorithmic}[1]
\Require Reasoning steps $S$, knowledge base $\text{KB}$
\Ensure Triadic fuzzy context $\mathcal{K} = (G, M, B, I)$

\State $G \leftarrow S$ \Comment{Objects are reasoning steps}
\State $M \leftarrow \text{DefineAttributes}()$ \Comment{Define logical properties}
\State $B \leftarrow \text{DefineConditions}(S, \text{KB})$ \Comment{Define uncertainty conditions}
\State Initialize $I: G \times M \times B \rightarrow [0,1]$ with zeros

\For{each $g \in G, m \in M, b \in B$}
    \State $I(g, m, b) \leftarrow \text{ComputeMembership}(g, m, b, \text{KB})$ \Comment{Calculate membership degree}
\EndFor

\State \Return{$(G, M, B, I)$}
\end{algorithmic}
\end{algorithm}

The membership degree computation is a critical component that leverages both the knowledge base and the LLM's internal confidence:

\begin{algorithm}[H]
\caption{ComputeMembership}
\label{alg:compute_membership}
\begin{algorithmic}[1]
\Require Reasoning step $g$, attribute $m$, condition $b$, knowledge base $\text{KB}$
\Ensure Membership degree $I(g, m, b) \in [0,1]$

\State $v_{\text{KB}} \leftarrow \text{KnowledgeBaseConsistency}(g, m, b, \text{KB})$ \Comment{Knowledge base consistency}
\State $v_{\text{LLM}} \leftarrow \text{ModelConfidence}(g, m, b)$ \Comment{Model's internal confidence}
\State $v_{\text{ASM}} \leftarrow \text{AssumptionValidity}(g, b, \text{KB})$ \Comment{Validity of assumptions under condition}

\State $I(g, m, b) \leftarrow \omega_1 \cdot v_{\text{KB}} + \omega_2 \cdot v_{\text{LLM}} + \omega_3 \cdot v_{\text{ASM}}$ \Comment{Weighted combination}

\State \Return{$I(g, m, b)$}
\end{algorithmic}
\end{algorithm}

The weights $\omega_1, \omega_2, \omega_3$ control the relative importance of different sources of evidence when computing membership degrees. These can be tuned based on the specific application domain and the reliability of different evidence sources.

\subsubsection{Concept Lattice Builder and Three-Way Decision Maker}

Once the triadic context is constructed, the Concept Lattice Builder creates the mathematical structure that supports uncertainty quantification. Algorithm \ref{alg:lattice_builder} outlines this process.

\begin{algorithm}[H]
\caption{ConceptLatticeBuilder}
\label{alg:lattice_builder}
\begin{algorithmic}[1]
\Require Triadic fuzzy context $\mathcal{K} = (G, M, B, I)$, threshold $\tau$
\Ensure Concept lattice $\mathcal{L}$

\State $\mathcal{L} \leftarrow \emptyset$ \Comment{Initialize empty lattice}
\State $\mathcal{C} \leftarrow \text{GenerateInitialConcepts}(\mathcal{K}, \tau)$ \Comment{Generate seed concepts}

\While{new concepts can be found}
    \State $c_{\text{new}} \leftarrow \text{NextConcept}(\mathcal{C}, \mathcal{L}, \mathcal{K}, \tau)$ \Comment{Find next concept}
    \If{$c_{\text{new}} \neq \text{null}$}
        \State $\mathcal{L} \leftarrow \mathcal{L} \cup \{c_{\text{new}}\}$ \Comment{Add to lattice}
        \State $\mathcal{L} \leftarrow \text{UpdateRelations}(\mathcal{L}, c_{\text{new}})$ \Comment{Update order relations}
    \Else
        \State \textbf{break} \Comment{No more concepts to find}
    \EndIf
\EndWhile

\State \Return{$\mathcal{L}$}
\end{algorithmic}
\end{algorithm}
% \begin{figure}[!t]
% \centering
% \begin{tikzpicture}[
%     box/.style={rectangle, draw, minimum width=3.5cm, minimum height=1.5cm, rounded corners, text centered, align=center},
%     green_box/.style={box, fill=green!20},
%     red_box/.style={box, fill=red!20},
%     yellow_box/.style={box, fill=yellow!20},
%     gray_box/.style={box, fill=gray!10}
% ]

% % Top concept
% \node[gray_box] (top) at (0,4) {Top Concept\\$(\{S_1,S_2,S_3\}, \emptyset, \{b_1,b_2\})$};

% % Middle concepts
% \node[green_box] (general) at (-3,2) {General Medicine\\$(\{S_1,S_2,S_3\}, \{m_1,m_2\}, \{b_1\})$\\ACCEPT};
% \node[red_box] (allergy) at (3,2) {Penicillin Allergy\\$(\{S_3\}, \{m_1,m_2\}, \{b_2\})$\\REJECT};

% % Lower concepts
% \node[green_box] (fever) at (-3,0) {Fever + Infection\\$(\{S_1,S_2\}, \{m_1,m_2\}, \{b_1\})$\\ACCEPT};
% \node[yellow_box] (antibiotics) at (0,0) {Antibiotics\\$(\{S_3\}, \{m_1,m_2\}, \{b_1\})$\\ACCEPT};

% % Bottom concept
% \node[gray_box] (bottom) at (0,-2) {Bottom Concept\\$(\emptyset, \{m_1,m_2,m_3\}, \{b_1,b_2\})$};

% % Arrows
% \draw[->] (top) -- (general);
% \draw[->] (top) -- (allergy);
% \draw[->] (general) -- (fever);
% \draw[->] (general) -- (antibiotics);
% \draw[->, dashed] (allergy) -- (antibiotics);
% \draw[->] (fever) -- (bottom);
% \draw[->] (antibiotics) -- (bottom);

% % Legend
% \node[draw, align=left] at (3,-1) {
%     Legend:\\
%     \rule{0.5cm}{0.3cm}\hspace{0.1cm} ACCEPT\\
%     \rule[0.15cm]{0.5cm}{0.3cm}\hspace{0.1cm} UNCERTAIN\\
%     \rule[0.3cm]{0.5cm}{0.3cm}\hspace{0.1cm} REJECT
% };

% \end{tikzpicture}
% \caption{Corrected medical decision lattice based on three-way formal concept analysis, showing how different contexts affect decision outcomes for medical statements.}
% \label{fig:medical_lattice_minimal}
% \end{figure}
\begin{figure}[!t]
    \centering
    \includegraphics[width=0.99\linewidth]{medical-decision-lattice.png}
\caption{Medical decision lattice incorporating context-sensitive three-way decisions. The structure now features:}
\label{fig:medical_lattice}
\end{figure}
The The lattice in \ref{fig:medical_lattice},  represents triadic concepts $(A, C, D)$ where $A \subseteq G = \{S_1, S_2, S_3\}$ (medical statements), $C \subseteq M = \{m_1, m_2, m_3\}$ (evaluation attributes), and $D \subseteq B = \{b_1, b_2\}$ (contexts). Decision outcomes (accept/reject/uncertain) are determined by comparing lower and upper membership bounds against thresholds $\alpha$ and $\beta$
The Three-Way Decision Maker then uses the concept lattice to make decisions about each reasoning step:

\begin{algorithm}[H]
\caption{ThreeWayDecision}
\label{alg:three_way_decision}
\begin{algorithmic}[1]
\Require Acceptance interval $[\mu^L_{\text{accept}}, \mu^U_{\text{accept}}]$, rejection interval $[\mu^L_{\text{reject}}, \mu^U_{\text{reject}}]$, thresholds $\alpha, \beta$
\Ensure Decision $d \in \{\text{accept}, \text{reject}, \text{abstain}\}$

\If{$\mu^L_{\text{accept}} \geq \alpha$}
    \State \Return{accept}
\ElsIf{$\mu^L_{\text{reject}} \geq \beta$}
    \State \Return{reject}
\Else
    \State \Return{abstain}
\EndIf
\end{algorithmic}
\end{algorithm}

\subsubsection{Uncertainty Resolution Module}

The Uncertainty Resolution Module is responsible for addressing steps with high uncertainty. Algorithm \ref{alg:uncertainty_resolver} details this process.

\begin{algorithm}[H]
\caption{UncertaintyResolver}
\label{alg:uncertainty_resolver}
\begin{algorithmic}[1]
\Require Uncertain steps $U$, all steps $S$, triadic context $\mathcal{K}$, knowledge base $\text{KB}$
\Ensure Updated reasoning steps $S'$

\State $S' \leftarrow S$ \Comment{Initialize with original steps}

\For{each uncertain step $u \in U$}
    \State $r_{\text{RAG}} \leftarrow \text{RAGRetriever}(u, \text{KB})$ \Comment{Try to resolve using retrieval}
    \If{$r_{\text{RAG}} \neq \text{null}$}
        \State Replace $u$ with $r_{\text{RAG}}$ in $S'$
        \State \textbf{continue} \Comment{Move to next uncertain step}
    \EndIf
    
    \State $\text{deps} \leftarrow \text{DependencyAnalysis}(u, S')$ \Comment{Find dependencies}
    \State $r_{\text{BT}} \leftarrow \text{Backtrack}(\text{deps}, u, \mathcal{K})$ \Comment{Try backtracking}
    \If{$r_{\text{BT}} \neq \text{null}$}
        \State Update dependencies and $u$ in $S'$ based on $r_{\text{BT}}$
    \Else
        \State Mark $u$ as explicitly uncertain in $S'$ \Comment{Cannot resolve uncertainty}
    \EndIf
\EndFor

\State \Return{$S'$}
\end{algorithmic}
\end{algorithm}
\begin{figure}[!t]
\centering
\begin{tikzpicture}[scale=0.8]
    % Draw the unit square
    \draw[thick] (0,0) rectangle (8,8);
    
    % Fill the regions with appropriate colors - strictly separated
    \fill[blue!30] (0,6) rectangle (6,8);
    \fill[red!30] (6,0) rectangle (8,6);
    \fill[yellow!30] (0,0) rectangle (6,6);
    
    % Label the axes with minimal text
    \node[below] at (4,0) {$\mu_{\text{no infection}}$};
    \node[left] at (0,4) {$\mu_{\text{infection}}$};
    
    % Draw threshold lines - simple labels
    \draw[dashed, thick] (0,6) -- (8,6) node[right] {$\alpha = 0.75$};
    \draw[dashed, thick] (6,0) -- (6,8) node[above] {$\beta = 0.75$};
    
    % Region labels - minimal text, well-spaced
    \node at (3,7) {Infection Region};
    \node at (7,3) {No Infection Region};
    \node at (3,1) {Abstention Region};
    
    % Example points - widely spaced
    \fill[black] (2,7) circle (0.15) node[right] {$S_3$};
    \fill[black] (7,2) circle (0.15) node[right] {$S_0$};
    \fill[black] (3,3) circle (0.15) node[right] {$S_2$};
    
    % Sub-regions - minimal dividers
    \draw[dashed, red] (3,0) -- (3,6);
    \draw[dashed, blue] (0,3) -- (6,3);
    
    % Key medical example information - positioned in empty spaces
    \node at (1.5,5) {$I(S_2, m_2, b_1) = 0.8$};
    \node at (4.5,2) {$I(S_2, m_2, b_2) = 0.6$};
    \node at (2,4) {$\tau = 0.7$};
\end{tikzpicture}
\caption{Medical diagnosis decision space showing infection vs. no-infection regions with threshold $\tau = 0.7$. Different confidence values are assigned in general medicine ($I(S_2, m_2, b_1) = 0.8$) versus pediatrics ($I(S_2, m_2, b_2) = 0.6$).}

\label{fig:decision_space}
\end{figure}
\ref{fig:decision_space} presents the decision space with sub-regions within the abstention area. The thresholds $\alpha = \beta = 0.75$ create clean separation between acceptance and rejection regions. The abstention region is further divided to indicate different types of uncertainty: low confidence, leaning toward acceptance, leaning toward rejection, and conflicting evidence.
The Uncertainty Resolution Module employs two main strategies for addressing uncertainty:

\begin{enumerate}
    \item \textbf{Retrieval-Augmented Generation (RAG)}: For steps with high uncertainty, the system attempts to retrieve relevant information from the knowledge base that could provide evidence for or against the reasoning step.
    
    \item \textbf{Dependency-Based Backtracking}: When RAG is insufficient, the system analyzes dependencies between reasoning steps and backtracks to revise earlier steps that may have led to the uncertainty.
\end{enumerate}

\begin{algorithm}[H]
\caption{RAGRetriever}
\label{alg:rag_retriever}
\begin{algorithmic}[1]
\Require Uncertain step $u$, knowledge base $\text{KB}$
\Ensure Resolved step $r$ or null

\State $q \leftarrow \text{FormulateQuery}(u)$ \Comment{Create query from uncertain step}
\State $E \leftarrow \text{RetrieveEvidence}(q, \text{KB})$ \Comment{Get evidence from knowledge base}

\If{$\text{RelevanceScore}(E, u) > \theta_{\text{rel}}$}
    \State $r \leftarrow \text{ReformulateStep}(u, E)$ \Comment{Reformulate with evidence}
    \State \Return{$r$}
\Else
    \State \Return{null} \Comment{No relevant evidence found}
\EndIf
\end{algorithmic}
\end{algorithm}

\begin{algorithm}[H]
\caption{Backtrack}
\label{alg:backtrack}
\begin{algorithmic}[1]
\Require Dependencies $\text{deps}$, uncertain step $u$, triadic context $\mathcal{K}$
\Ensure Revised steps or null

\State Sort $\text{deps}$ by influence on $u$
\For{each dependency $d \in \text{deps}$}
    \State $A_d \leftarrow \text{ExtractAssumptions}(d)$ \Comment{Get assumptions in dependency}
    \For{each assumption $a \in A_d$}
        \State $v_a \leftarrow \text{ValidateAssumption}(a, \mathcal{K})$ \Comment{Check assumption validity}
        \If{$v_a < \theta_{\text{valid}}$}
            \State $d' \leftarrow \text{ReviseStep}(d, a)$ \Comment{Revise step with corrected assumption}
            \State $u' \leftarrow \text{PropagateRevision}(d', u)$ \Comment{Update uncertain step}
            \State \Return{$\{d', u'\}$}
        \EndIf
    \EndFor
\EndFor

\State \Return{null} \Comment{Cannot revise through backtracking}
\end{algorithmic}
\end{algorithm}

\subsection{Decision Aggregation}

The final stage of the 3WayCOT algorithm aggregates the decisions and uncertainty measures across all reasoning steps to produce an overall answer with uncertainty quantification:

\begin{algorithm}[H]
\caption{Aggregator}
\label{alg:aggregator}
\begin{algorithmic}[1]
\Require Reasoning steps $S$, decisions $\{d_i\}$, acceptance intervals $\{[\mu^L_{\text{accept}}, \mu^U_{\text{accept}}]_i\}$, rejection intervals $\{[\mu^L_{\text{reject}}, \mu^U_{\text{reject}}]_i\}$
\Ensure Answer $y$ with uncertainty bounds

\State $S_{\text{accept}} \leftarrow \{S_i \in S \mid d_i = \text{accept}\}$ \Comment{Accepted steps}
\State $S_{\text{reject}} \leftarrow \{S_i \in S \mid d_i = \text{reject}\}$ \Comment{Rejected steps}
\State $S_{\text{abstain}} \leftarrow \{S_i \in S \mid d_i = \text{abstain}\}$ \Comment{Uncertain steps}

\If{$|S_{\text{reject}}| > 0$}
    \State $y \leftarrow \text{GenerateAlternativeAnswer}(S, S_{\text{reject}})$ \Comment{Answer addressing rejections}
\ElsIf{$|S_{\text{abstain}}| > \theta_{\text{abs}} \cdot |S|$}
    \State $y \leftarrow \text{GenerateCaveatedAnswer}(S, S_{\text{abstain}})$ \Comment{Answer with caveats}
\Else
    \State $y \leftarrow \text{GenerateConfidentAnswer}(S_{\text{accept}})$ \Comment{Answer based on accepted steps}
\EndIf

\State $[\mu^L_y, \mu^U_y] \leftarrow \text{PropagateUncertainty}(S, \{[\mu^L_{\text{accept}}, \mu^U_{\text{accept}}]_i\}, \{[\mu^L_{\text{reject}}, \mu^U_{\text{reject}}]_i\})$
\State $y \leftarrow \text{AnnotateWithUncertainty}(y, [\mu^L_y, \mu^U_y])$ \Comment{Add uncertainty bounds to answer}

\State \Return{$y$}
\end{algorithmic}
\end{algorithm}

This final step ensures that the system's output reflects the uncertainty in its reasoning process, providing users with both an answer and a characterization of the confidence in that answer.

\begin{example}[Medical Diagnosis Aggregation]
For the medical diagnosis example:

\begin{itemize}
    \item $S_1$: "Patient has fever of 102°F" - \textbf{accept} with $[0.9, 0.95]$
    \item $S_2$: "Patient likely has bacterial infection" - \textbf{accept} with $[0.7, 0.85]$
    \item $S_3$: "Recommend antibiotics treatment" - \textbf{abstain} with $[0.5, 0.7]$ (in general context) or \textbf{reject} with $[0.2, 0.3]$ (with penicillin allergy)
\end{itemize}

The aggregated answer might be: "The patient's fever of 102°F suggests a bacterial infection [confidence: 0.7-0.85]. Consider antibiotics treatment [confidence: 0.5-0.7], but first confirm patient has no penicillin allergy, as this would contraindicate this treatment [confidence: 0.7-0.8]."

This answer explicitly communicates the system's confidence in different aspects of the diagnosis and highlights critical uncertainty around the treatment recommendation.
\end{example}

\subsection{Complexity Analysis}

The time and space complexity of the 3WayCOT algorithm is determined primarily by the construction of the triadic concept lattice, which dominates the computational cost.

\begin{theorem}[Computational Complexity]
The worst-case time complexity of Algorithm~\ref{alg:3waycot} is $O(|S| \cdot |G|^2 \cdot |M| \cdot |B|)$, and the worst-case space complexity is $O(|S| \cdot |G| \cdot |M| \cdot |B| + |\mathcal{L}|)$, where $|\mathcal{L}|$ is the size of the concept lattice.
\end{theorem}

\begin{proof}
The complexity of the main algorithm is dominated by the following operations:
\begin{itemize}
    \item For each of the $|S|$ reasoning steps, we construct a triadic context and build a concept lattice.
    \item Building the triadic concept lattice (Algorithm~\ref{alg:concept_lattice}) has a worst-case time complexity of $O(|G|^2 \cdot |M| \cdot |B|)$, as we need to check all possible combinations of objects, attributes, and conditions.
    \item The membership calculation (Algorithm~\ref{alg:membership}) has a time complexity of $O(|\mathcal{L}| \cdot |M| \cdot |B|)$, where $|\mathcal{L}|$ is the number of concepts in the lattice.
    \item The uncertainty resolution (Algorithm~\ref{alg:uncertainty}) has a time complexity of $O(|K| \cdot log|K| + |K| \cdot |S_i|)$, where $|K|$ is the size of the knowledge base and $|S_i|$ is the length of the reasoning step.
\end{itemize}

In the worst case, $|\mathcal{L}| = 2^{|G|}$, but in practice, the number of concepts is much smaller due to the constraints imposed by the incidence relation. The space complexity is dominated by storing the concept lattice for each reasoning step, which requires $O(|S| \cdot |\mathcal{L}|)$ space.

Therefore, the overall worst-case time complexity is $O(|S| \cdot |G|^2 \cdot |M| \cdot |B|)$, and the worst-case space complexity is $O(|S| \cdot |G| \cdot |M| \cdot |B| + |\mathcal{L}|)$.
\end{proof} 
\section{Experimental Evaluation}

In this section, we present a comprehensive evaluation of 3WayCOT across multiple datasets, comparing its performance to several baseline methods. We also provide ablation studies to analyze the contribution of each component of our framework.

\subsection{Datasets and Experimental Setup}

\subsubsection{Datasets}

We evaluate 3WayCOT on three distinct datasets designed to test different aspects of uncertainty-aware reasoning:

\begin{itemize}
    \item \textbf{MedDiag:} A medical diagnosis dataset containing 500 cases with symptoms, possible diagnoses, and ground truth labels. Each case includes patient symptoms, laboratory test results, and the correct diagnosis verified by medical professionals. We specifically selected cases with potentially ambiguous symptoms that could lead to multiple possible diagnoses.
    
    \item \textbf{LogicQA:} A logical reasoning benchmark with 300 multi-step deduction problems, where each problem requires a sequence of logical inferences to arrive at the correct conclusion. The problems include both clear-cut cases and deliberately ambiguous scenarios.
    
    \item \textbf{AmbigNLI:} A natural language inference dataset containing 200 deliberately ambiguous premise-hypothesis pairs. Unlike standard NLI datasets, AmbigNLI includes cases where the relationship between premise and hypothesis is genuinely uncertain given the provided information.
\end{itemize}

Table~\ref{tab:dataset_stats} provides the key statistics of these datasets.

\begin{table}[!t]
\centering
\caption{Dataset Statistics}
\label{tab:dataset_stats}
\begin{tabular}{lccc}
\toprule
\textbf{Dataset} & \textbf{Samples} & \textbf{Avg. Length} & \textbf{Ambiguity Rate (\%)} \\
\midrule
MedDiag & 500 & 157.3 words & 32.4 \\
LogicQA & 300 & 96.5 words & 25.7 \\
AmbigNLI & 200 & 68.2 words & 40.0 \\
\bottomrule
\end{tabular}
\end{table}

The ambiguity rate represents the percentage of samples that contain inherent uncertainty, as determined by human expert annotators. These cases require additional information or context beyond what is provided in the problem statement to reach a definitive conclusion.

\subsubsection{Baseline Methods}

We compare 3WayCOT against several baseline methods:

\begin{itemize}
    \item \textbf{Vanilla:} Direct query-to-answer mapping without explicit reasoning steps.
    \item \textbf{CoT:} Standard chain-of-thought prompting with the prefix "Let's think step by step".
    \item \textbf{Self-Consistency (SC):} Multiple CoT paths with majority voting for the final answer.
    \item \textbf{RAG-CoT:} Chain-of-thought enhanced with retrieval-augmented generation.
\end{itemize}

All methods use the same underlying language model (a fine-tuned variant of GPT-4) to ensure fair comparison. For self-consistency, we generate 5 reasoning paths for each query.

\subsubsection{Evaluation Metrics}

We evaluate the methods using the following metrics:

\begin{itemize}
    \item \textbf{Accuracy:} The percentage of correct answers according to ground truth.
    \item \textbf{Uncertainty Resolution Rate:} The percentage of initially uncertain reasoning steps that are successfully resolved.
    \item \textbf{Hallucination Rate:} The percentage of generated statements that contradict established facts or ground truth.
    \item \textbf{Calibration Error:} The mean absolute difference between predicted confidence and empirical accuracy.
    \item \textbf{F1 Score:} The harmonic mean of precision and recall for multi-class classification tasks.
\end{itemize}

\subsubsection{Implementation Details}

We implemented 3WayCOT using Python 3.9 with PyTorch 2.0 for the underlying models. For the triadic FCA implementation, we developed a custom library based on the concepts-py package. The hyperparameters were selected based on performance on a validation set (20\% of the data), with the final values set as follows:

\begin{itemize}
    \item Acceptance threshold $\alpha = 0.7$
    \item Rejection threshold $\beta = 0.6$
    \item Similarity threshold $\tau = 0.5$
    \item Uncertainty resolution threshold $\theta = 0.85$
\end{itemize}

All experiments were run on a server with 8 NVIDIA A100 GPUs and 512GB RAM. For reproducibility, we set the random seed to 42 across all experiments.

\subsection{Experimental Results}
