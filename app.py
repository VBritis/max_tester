import streamlit as st
import openai
import json
import pandas as pd

from llm_core import (
    extract_schema,
    generate_tests,
    validate_errors,
)

st.set_page_config(page_title="Max Tester", page_icon="🧪", layout="wide")

# ── Session State Defaults ──────────────────────────────────────────────────

DEFAULTS = {
    "step": 1,
    "context": "",
    "examples": "",
    "num_tests": 50,
    "extracted_schema": None,
    "preview_tests": None,
    "final_tests": None,
    "validation_results": None,
    "quick_validation_results": None,
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("🧪 Max Tester")
st.sidebar.caption("Pipeline de testes automatizados para LLMs")
st.sidebar.divider()

api_key = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="sk-proj-...")

if not api_key:
    st.info("Insira sua API Key da OpenAI na barra lateral para começar.")
    st.stop()

client = openai.OpenAI(api_key=api_key)


# ── Tabs ────────────────────────────────────────────────────────────────────

tab_pipeline, tab_validator, tab_refiner = st.tabs([
    "🔬 Pipeline Completo",
    "🔍 Validação Rápida",
    "✨ Prompt Refiner",
])


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1 — Pipeline Completo (fluxo de 5 steps)
# ═══════════════════════════════════════════════════════════════════════════

with tab_pipeline:

    STEP_LABELS = {
        1: "Configurar Inputs",
        2: "Revisar Schema",
        3: "Validar Preview",
        4: "Testes Gerados",
        5: "Resultados da Validação",
    }

    st.progress(st.session_state.step / 5)
    st.caption(f"Etapa {st.session_state.step} de 5 — {STEP_LABELS[st.session_state.step]}")
    st.divider()

    # ── Step 1: Input ───────────────────────────────────────────────────────

    if st.session_state.step == 1:
        st.header("1. Configure os inputs")

        st.session_state.context = st.text_area(
            "Contexto / Direcionamento",
            value=st.session_state.context,
            height=120,
            placeholder="Descreva o que seu LLM faz, o domínio, instruções específicas...",
        )

        st.session_state.examples = st.text_area(
            "Exemplos da Estrutura (logs de input/output)",
            value=st.session_state.examples,
            height=250,
            placeholder=(
                'User: "Faz um pix de 150 reais para o João"\n'
                'Output: {"tool_name": "payment_gateway", "amount": 150.00, ...}\n\n'
                'User: "Ve meu saldo"\n'
                'Output: {"tool_name": "transactions", "resource_type": "view_invoice"}'
            ),
        )

        st.session_state.num_tests = st.number_input(
            "Quantidade de testes para produção em massa",
            min_value=10,
            max_value=500,
            value=st.session_state.num_tests,
            step=10,
        )

        if st.button("Extrair Schema →", type="primary", use_container_width=True, key="btn_extract"):
            if not st.session_state.examples.strip():
                st.warning("Cole pelo menos alguns exemplos de logs.")
            else:
                with st.spinner("Extraindo schema dos exemplos..."):
                    try:
                        schema = extract_schema(
                            client,
                            st.session_state.context,
                            st.session_state.examples,
                        )
                        st.session_state.extracted_schema = schema
                        st.session_state.step = 2
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao extrair schema: {e}")

    # ── Step 2: Schema Review ───────────────────────────────────────────────

    elif st.session_state.step == 2:
        st.header("2. Revise o Schema Extraído")

        schema = st.session_state.extracted_schema

        for payload in schema.structure.payloads:
            st.subheader(f"Payload: `{payload.name}`")
            rows = []
            for node in payload.nodes:
                rows.append({
                    "Campo": node.field,
                    "Fixo?": "✓" if node.is_fixed else "✗",
                    "Valor Fixo": str(node.fixed_value) if node.fixed_value is not None else "—",
                    "Type Hint": node.type_hint,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with st.expander("Ver JSON completo"):
            st.json(json.loads(schema.model_dump_json()))

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Voltar", use_container_width=True, key="btn_back_2"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button("Aprovar & Gerar Preview →", type="primary", use_container_width=True, key="btn_preview"):
                with st.spinner("Gerando preview (5 exemplos)..."):
                    try:
                        preview = generate_tests(client, schema, count=5)
                        st.session_state.preview_tests = preview
                        st.session_state.step = 3
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao gerar preview: {e}")

    # ── Step 3: Preview Validation ──────────────────────────────────────────

    elif st.session_state.step == 3:
        st.header("3. Valide o Preview dos Testes")

        preview = st.session_state.preview_tests

        for i, test in enumerate(preview.response, 1):
            with st.expander(f"Teste {i}: {test.user_input[:80]}"):
                st.markdown(f"**User Input:** {test.user_input}")
                st.markdown(f"**Payload:** `{test.expected_payload.name}`")
                payload_data = {node.field: node.value for node in test.expected_payload.nodes}
                st.json(payload_data)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Rejeitar & Voltar", use_container_width=True, key="btn_reject"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            n = st.session_state.num_tests
            if st.button(f"Aprovar & Gerar {n} Testes →", type="primary", use_container_width=True, key="btn_mass"):
                with st.spinner(f"Gerando {n} testes..."):
                    try:
                        final = generate_tests(
                            client,
                            st.session_state.extracted_schema,
                            count=n,
                        )
                        st.session_state.final_tests = final
                        st.session_state.step = 4
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao gerar testes: {e}")

    # ── Step 4: Generated Tests + Error Log Input ───────────────────────────

    elif st.session_state.step == 4:
        st.header("4. Testes Gerados")

        final = st.session_state.final_tests

        rows = []
        for test in final.response:
            payload_data = {node.field: node.value for node in test.expected_payload.nodes}
            rows.append({
                "User Input": test.user_input,
                "Payload": test.expected_payload.name,
                **payload_data,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        json_str = final.model_dump_json(indent=2)
        st.download_button(
            "⬇ Download Testes (JSON)",
            data=json_str,
            file_name="test_cases.json",
            mime="application/json",
            use_container_width=True,
        )

        st.divider()
        st.subheader("Cole os logs de erro")
        st.caption(
            "Rode os testes no seu sistema e cole aqui **somente os erros**. "
            "Formato: JSON array com objetos contendo `user_input`, `expected_payload` e `actual_payload`."
        )

        error_logs = st.text_area(
            "Error Logs (JSON)",
            height=250,
            key="pipeline_error_logs",
            placeholder=json.dumps([
                {
                    "user_input": "Faz um pix de 100 pro João",
                    "expected_payload": {"name": "payment_gateway", "amount": 100},
                    "actual_payload": {"name": "transactions", "amount": 100},
                }
            ], indent=2, ensure_ascii=False),
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Voltar", use_container_width=True, key="btn_back_4"):
                st.session_state.step = 3
                st.rerun()
        with col2:
            if st.button("Validar Erros →", type="primary", use_container_width=True, key="btn_validate"):
                if not error_logs.strip():
                    st.warning("Cole os logs de erro antes de validar.")
                else:
                    with st.spinner("Validando erros..."):
                        try:
                            results = validate_errors(client, error_logs)
                            st.session_state.validation_results = results
                            st.session_state.step = 5
                            st.rerun()
                        except ValueError as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(f"Erro na validação: {e}")

    # ── Step 5: Validation Results ──────────────────────────────────────────

    elif st.session_state.step == 5:
        st.header("5. Resultados da Validação")

        results = st.session_state.validation_results
        total = len(results)
        test_faults = sum(1 for r in results if r["validation"].is_test_flawed)
        llm_faults = sum(1 for r in results if r["validation"].is_llm_flawed)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Erros", total)
        col2.metric("Culpa do Teste", f"{test_faults} ({100 * test_faults // max(total, 1)}%)")
        col3.metric("Culpa do LLM", f"{llm_faults} ({100 * llm_faults // max(total, 1)}%)")

        st.divider()

        for i, r in enumerate(results, 1):
            v = r["validation"]
            label = "🟠 Teste" if v.is_test_flawed else "🔴 LLM"
            with st.expander(f"Erro {i} — {label}: {r['error']['user_input'][:60]}"):
                st.markdown(f"**Diagnóstico:** {v.diagnosis}")
                st.markdown(f"**Ação Sugerida:** {v.suggested_action}")
                st.divider()
                st.markdown("**User Input:**")
                st.code(r["error"]["user_input"])
                st.markdown("**Expected Payload:**")
                st.json(r["error"]["expected_payload"])
                st.markdown("**Actual Payload:**")
                st.json(r["error"]["actual_payload"])

        if st.button("🔄 Recomeçar", type="primary", use_container_width=True, key="btn_restart"):
            for key, val in DEFAULTS.items():
                st.session_state[key] = val
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2 — Validação Rápida (standalone, sem pipeline)
# ═══════════════════════════════════════════════════════════════════════════

with tab_validator:
    st.header("🔍 Validação Rápida")
    st.caption(
        "Já tem os logs de erro? Cole direto aqui — sem precisar passar pelo pipeline de geração."
    )

    st.divider()

    error_logs_quick = st.text_area(
        "Error Logs (JSON)",
        height=300,
        key="quick_error_logs",
        placeholder=json.dumps([
            {
                "user_input": "Faz um pix de 100 pro João",
                "expected_payload": {"name": "payment_gateway", "amount": 100.00, "receiver": "João"},
                "actual_payload": {"name": "transactions", "amount": 100.00, "receiver": "João"},
            },
            {
                "user_input": "Quero ver minha fatura de fevereiro",
                "expected_payload": {"name": "transactions", "resource_type": "view_invoice", "date": "02-2025"},
                "actual_payload": {"name": "transactions", "resource_type": "view_invoice", "date": None},
            },
        ], indent=2, ensure_ascii=False),
    )

    st.info(
        "**Formato esperado:** JSON array onde cada objeto tem:\n"
        "- `user_input` — o que o usuário mandou\n"
        "- `expected_payload` — o que deveria ter saído\n"
        "- `actual_payload` — o que o LLM realmente retornou"
    )

    if st.button("Validar →", type="primary", use_container_width=True, key="btn_quick_validate"):
        if not error_logs_quick.strip():
            st.warning("Cole os logs de erro antes de validar.")
        else:
            with st.spinner("Analisando erros..."):
                try:
                    results = validate_errors(client, error_logs_quick)
                    st.session_state.quick_validation_results = results
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Erro na validação: {e}")

    # Resultados
    if st.session_state.quick_validation_results:
        st.divider()
        results = st.session_state.quick_validation_results
        total = len(results)
        test_faults = sum(1 for r in results if r["validation"].is_test_flawed)
        llm_faults = sum(1 for r in results if r["validation"].is_llm_flawed)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Erros", total)
        col2.metric("Culpa do Teste", f"{test_faults} ({100 * test_faults // max(total, 1)}%)")
        col3.metric("Culpa do LLM", f"{llm_faults} ({100 * llm_faults // max(total, 1)}%)")

        st.divider()

        for i, r in enumerate(results, 1):
            v = r["validation"]
            label = "🟠 Teste" if v.is_test_flawed else "🔴 LLM"
            with st.expander(f"Erro {i} — {label}: {r['error']['user_input'][:60]}"):
                st.markdown(f"**Diagnóstico:** {v.diagnosis}")
                st.markdown(f"**Ação Sugerida:** {v.suggested_action}")
                st.divider()
                st.markdown("**User Input:**")
                st.code(r["error"]["user_input"])
                st.markdown("**Expected Payload:**")
                st.json(r["error"]["expected_payload"])
                st.markdown("**Actual Payload:**")
                st.json(r["error"]["actual_payload"])

        if st.button("Limpar resultados", use_container_width=True, key="btn_clear_quick"):
            st.session_state.quick_validation_results = None
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 3 — Prompt Refiner (placeholder)
# ═══════════════════════════════════════════════════════════════════════════

with tab_refiner:

    # Hero section
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 1rem;">
            <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">✨ Prompt Refiner</h1>
            <p style="font-size: 1.2rem; color: #888; max-width: 600px; margin: 0 auto;">
                Receba sugestões inteligentes para melhorar o prompt do seu sistema
                com base nos erros detectados pelo Validator.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Feature cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            #### 🎯 Análise de Falhas
            Identifica padrões nos erros do LLM e mapeia
            quais regras estão faltando no prompt.
            """
        )

    with col2:
        st.markdown(
            """
            #### 🔧 Sugestões de Rewrite
            Gera trechos de prompt otimizados que você pode
            copiar e testar diretamente.
            """
        )

    with col3:
        st.markdown(
            """
            #### 📊 Before / After
            Compara o comportamento do LLM antes e depois
            das mudanças sugeridas.
            """
        )

    st.divider()

    st.markdown(
        """
        <div style="text-align: center; padding: 2rem;
                    border: 2px dashed #444; border-radius: 12px; margin: 1rem 0;">
            <p style="font-size: 1.1rem; color: #666;">
                🚧 Em desenvolvimento — em breve disponível.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
