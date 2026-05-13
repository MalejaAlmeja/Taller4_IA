from __future__ import annotations

from planning.pddl import ActionSchema, State, Objects, get_applicable_actions, apply_action


def nullHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """Trivial heuristic — always returns 0 (equivalent to uniform-cost search)."""
    return 0


# ---------------------------------------------------------------------------
# Punto 4a – Ignore-Preconditions Heuristic
# ---------------------------------------------------------------------------


def ignorePreconditionsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the number of actions needed to satisfy all goal fluents,
    ignoring all action preconditions.

    With no preconditions, any action can be applied at any time.
    Each action can satisfy all goal fluents in its add_list in one step.
    The minimum number of actions to cover all unsatisfied goal fluents is
    a lower bound on the true plan length → this heuristic is admissible.

    Algorithm (greedy set cover):
      1. Compute unsatisfied = goal − state  (fluents still needed).
      2. Ground all actions ignoring preconditions and collect their add_lists.
      3. Greedily pick the action whose add_list covers the most unsatisfied fluents.
      4. Repeat until all fluents are covered; count the actions used.

    Tip: frozenset supports set difference (-) and intersection (&).
         You only need to ground actions once per call (use get_applicable_actions
         with the initial state, or generate all groundings regardless of state).
         Remember: with no preconditions, every grounding is "applicable".
    """

    fluentes_estado = state.fluents if hasattr(state, 'fluents') else state
    fluentes_objetivo = goal.fluents if hasattr(goal, 'fluents') else goal

    fluentes_faltantes = fluentes_objetivo - fluentes_estado

    if not fluentes_faltantes:
        return 0

    todos_los_fluentes = fluentes_estado | fluentes_objetivo
    estado_falso = State(todos_los_fluentes)

    acciones_disponibles = get_applicable_actions(estado_falso, domain, objects)

    listas_de_efectos = [frozenset(accion.add_list) for accion in acciones_disponibles]

    conteo = 0
    por_cubrir = frozenset(fluentes_faltantes)

    while por_cubrir:
        mejor_cobertura = frozenset()

        for efectos in listas_de_efectos:
            cobertura = efectos & por_cubrir 
            if len(cobertura) > len(mejor_cobertura):
                mejor_cobertura = cobertura

        if not mejor_cobertura:
            return float('inf')

        por_cubrir -= mejor_cobertura
        conteo += 1

    return conteo


# ---------------------------------------------------------------------------
# Punto 4b – Ignore-Delete-Lists Heuristic
# ---------------------------------------------------------------------------


def ignoreDeleteListsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the plan cost by solving a relaxed problem where no action
    has a delete list (effects never remove fluents from the state).

    In this monotone relaxation, the state only grows over time (fluents are
    never removed), so hill-climbing always makes progress and cannot loop.

    Algorithm (hill-climbing on the relaxed problem):
      1. Start from the current state with a relaxed (monotone) apply function.
      2. At each step, pick the grounded action that adds the most unsatisfied
         goal fluents (greedy hill-climbing).
      3. Count steps until all goal fluents are satisfied (or until no progress).

    Tip: In the relaxed problem, apply_action never removes fluents.
         You can implement this by treating del_list as empty for all actions.
         Use get_applicable_actions to enumerate applicable grounded actions at
         each step (preconditions still apply in the relaxed model).
    """
    fluentes_estado = state.fluents if hasattr(state, 'fluents') else state
    fluentes_objetivo = goal.fluents if hasattr(goal, 'fluents') else goal

    fluentes_actuales = set(fluentes_estado)
    pasos = 0
    limite = 1000  

    for _ in range(limite):
        fluentes_faltantes = fluentes_objetivo - fluentes_actuales

        if not fluentes_faltantes:
            return pasos

        estado_actual = State(frozenset(fluentes_actuales))
        acciones_aplicables = get_applicable_actions(estado_actual, domain, objects)

        if not acciones_aplicables:
            return float('inf')  
        
        mejor_accion = None
        mejor_ganancia = 0

        for accion in acciones_aplicables:
            ganancia = len(frozenset(accion.add_list) & fluentes_faltantes)
            if ganancia > mejor_ganancia:
                mejor_ganancia = ganancia
                mejor_accion = accion

        if mejor_accion is None or mejor_ganancia == 0:
            return float('inf')

        fluentes_actuales |= set(mejor_accion.add_list)
        pasos += 1

    return float('inf') 