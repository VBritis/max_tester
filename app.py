import streamlit as st
import openai
import json
import pandas as pd

from llm_core import (
    extract_schema,
    generate_tests,
    validate_errors,
    refine_prompt,
    padronizer
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
    "refiner_result": None,
    "refiner_changes": None,
    "refiner_analysis": None,
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
            placeholder="Descreva o que seu LLM faz, o domínio, instruções específicas, formatação que deseja dos dados (gírias, erros de digitação, etc)...",
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
                        preview = generate_tests(client, schema, count=5, context = st.session_state.context)
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
                            context = st.session_state.context
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
        final_response = padronizer(final)

        json_str = json.dumps(final_response, indent=2, ensure_ascii=False)
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
            "Rode os testes no seu sistema e cole aqui **somente os erros**."
        )

        log_format = st.radio(
            "Formato dos logs",
            ["JSON", "CSV", "Logs Brutos"],
            horizontal=True,
            key="pipeline_log_format",
        )

        placeholders = {
            "JSON": json.dumps([
                {
                    "user_input": "Faz um pix de 100 pro João",
                    "expected_payload": {"name": "payment_gateway", "amount": 100},
                    "actual_payload": {"name": "transactions", "amount": 100},
                }
            ], indent=2, ensure_ascii=False),
            "CSV": (
                "user_input,expected_payload,actual_payload\n"
                '"Faz um pix de 100 pro João","{""name"":""payment_gateway"",""amount"":100}","{""name"":""transactions"",""amount"":100}"'
            ),
            "Logs Brutos": (
                '[2024-05-20 10:15:32] EVENT: execution_trace\n'
                'USER_PROMPT: "Faz um pix de 100 pro João"\n'
                'EXPECTED_BEHAVIOR: Call tool "payment_gateway" with amount=100\n'
                'MODEL_EXECUTION: Called "transactions" with amount=100'
            ),
        }

        error_logs = st.text_area(
            "Error Logs",
            height=250,
            key="pipeline_error_logs",
            placeholder=placeholders[log_format],
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
                    fmt_map = {"JSON": "json", "CSV": "csv", "Logs Brutos": "raw"}
                    fmt = fmt_map[log_format]
                    spinner_msg = "Estruturando logs brutos e validando..." if fmt == "raw" else "Validando erros..."
                    with st.spinner(spinner_msg):
                        try:
                            results = validate_errors(client, error_logs, fmt=fmt)
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

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Recomeçar", type="primary", use_container_width=True, key="btn_restart"):
                for key, val in DEFAULTS.items():
                    st.session_state[key] = val
                st.rerun()
        with col2:
            if llm_faults > 0:
                # Coleta as análises dos erros de LLM para alimentar o Prompt Refiner
                llm_analyses = "\n\n".join(
                    f"- Input: {r['error']['user_input']}\n  Diagnóstico: {r['validation'].diagnosis}\n  Ação: {r['validation'].suggested_action}"
                    for r in results if r["validation"].is_llm_flawed
                )
                if st.button("✨ Ir para Prompt Refiner →", use_container_width=True, key="btn_go_refiner"):
                    st.session_state.refiner_analysis = llm_analyses
                    st.session_state.refiner_analysis_input = llm_analyses
                    st.toast("Análise copiada! Clique na aba ✨ Prompt Refiner para continuar.", icon="✨")
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

    quick_log_format = st.radio(
        "Formato dos logs",
        ["JSON", "CSV", "Logs Brutos"],
        horizontal=True,
        key="quick_log_format",
    )

    quick_placeholders = {
        "JSON": json.dumps([
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
        "CSV": (
            "user_input,expected_payload,actual_payload\n"
            '"Faz um pix de 100 pro João","{""name"":""payment_gateway"",""amount"":100}","{""name"":""transactions"",""amount"":100}"'
        ),
        "Logs Brutos": (
            '[2024-05-20 10:15:32] EVENT: execution_trace\n'
            'USER_PROMPT: "Faz um pix de 100 pro João"\n'
            'EXPECTED_BEHAVIOR: Call tool "payment_gateway" with amount=100\n'
            'MODEL_EXECUTION: Called "transactions" with amount=100'
        ),
    }

    error_logs_quick = st.text_area(
        "Error Logs",
        height=300,
        key="quick_error_logs",
        placeholder=quick_placeholders[quick_log_format],
    )

    format_hints = {
        "JSON": (
            "**Formato esperado:** JSON array onde cada objeto tem:\n"
            "- `user_input` — o que o usuário mandou\n"
            "- `expected_payload` — o que deveria ter saído\n"
            "- `actual_payload` — o que o LLM realmente retornou"
        ),
        "CSV": (
            "**Formato esperado:** CSV com colunas `user_input`, `expected_payload`, `actual_payload`.\n"
            "Os payloads devem ser JSON válido dentro das células."
        ),
        "Logs Brutos": (
            "**Cole os logs brutos** de execução do seu sistema. "
            "Uma LLM irá analisar e estruturar automaticamente antes da validação."
        ),
    }
    st.info(format_hints[quick_log_format])

    if st.button("Validar →", type="primary", use_container_width=True, key="btn_quick_validate"):
        if not error_logs_quick.strip():
            st.warning("Cole os logs de erro antes de validar.")
        else:
            fmt_map = {"JSON": "json", "CSV": "csv", "Logs Brutos": "raw"}
            fmt = fmt_map[quick_log_format]
            spinner_msg = "Estruturando logs brutos e validando..." if fmt == "raw" else "Analisando erros..."
            with st.spinner(spinner_msg):
                try:
                    results = validate_errors(client, error_logs_quick, fmt=fmt)
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

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Limpar resultados", use_container_width=True, key="btn_clear_quick"):
                st.session_state.quick_validation_results = None
                st.rerun()
        with col2:
            if llm_faults > 0:
                llm_analyses_quick = "\n\n".join(
                    f"- Input: {r['error']['user_input']}\n  Diagnóstico: {r['validation'].diagnosis}\n  Ação: {r['validation'].suggested_action}"
                    for r in results if r["validation"].is_llm_flawed
                )
                if st.button("✨ Ir para Prompt Refiner →", use_container_width=True, key="btn_go_refiner_quick"):
                    st.session_state.refiner_analysis = llm_analyses_quick
                    st.session_state.refiner_analysis_input = llm_analyses_quick
                    st.toast("Análise copiada! Clique na aba ✨ Prompt Refiner para continuar.", icon="✨")
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 3 — Prompt Refiner (placeholder)
# ═══════════════════════════════════════════════════════════════════════════

with tab_refiner:
    st.header("✨ Prompt Refiner")
    st.caption(
        "Receba sugestões inteligentes para melhorar o prompt do seu sistema "
        "com base nos erros detectados pelo Validator."
    )

    st.divider()

    old_prompt = st.text_area(
        "Prompt Atual do seu sistema",
        height=200,
        key="refiner_old_prompt",
        placeholder="Cole aqui o prompt atual do sistema que você quer melhorar...",
    )

    analysis_input = st.text_area(
        "Análise de Melhoria",
        height=200,
        key="refiner_analysis_input",
        placeholder=(
            "Cole aqui o feedback/análise dos erros encontrados, ou use o botão "
            "'Ir para Prompt Refiner' na aba de validação para preencher automaticamente."
        ),
    )

    if st.session_state.refiner_analysis:
        st.success("Análise preenchida automaticamente a partir da validação de erros do LLM.")

    if st.button("Refinar Prompt →", type="primary", use_container_width=True, key="btn_refine"):
        if not old_prompt.strip():
            st.warning("Cole o prompt atual antes de refinar.")
        elif not analysis_input.strip():
            st.warning("Cole a análise de melhoria antes de refinar.")
        else:
            with st.spinner("Refinando prompt..."):
                try:
                    refined, changes = refine_prompt(client, old_prompt, analysis_input)
                    st.session_state.refiner_result = refined
                    st.session_state.refiner_changes = changes
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao refinar prompt: {e}")

    if st.session_state.refiner_result:
        st.divider()

        # ── Resumo das mudanças (destaque principal) ──
        st.subheader("📋 Resumo das Mudanças")
        st.info(st.session_state.refiner_changes)

        # ── Prompt refinado (colapsado por padrão) ──
        with st.expander("Ver Prompt Refinado completo", expanded=False):
            st.code(st.session_state.refiner_result, language="markdown")

            st.download_button(
                "⬇ Download Prompt Refinado",
                data=st.session_state.refiner_result,
                file_name="refined_prompt.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # ── Input de contextualização extra para re-refinar ──
        st.divider()
        st.subheader("🔄 Ajustar Refinamento")
        st.caption("Não ficou como esperava? Adicione contexto extra e refine novamente.")

        extra_context = st.text_area(
            "Contextualização adicional",
            height=120,
            key="refiner_extra_context",
            placeholder="Ex: 'O prompt precisa lidar também com pedidos em espanhol' ou 'Não remova a seção de few-shots'...",
        )

        if st.button("Re-refinar com contexto →", type="primary", use_container_width=True, key="btn_rerefine"):
            if not extra_context.strip():
                st.warning("Escreva alguma contextualização adicional antes de re-refinar.")
            else:
                combined_analysis = (
                    f"{analysis_input}\n\n"
                    f"[CONTEXTUALIZAÇÃO ADICIONAL DO USUÁRIO]\n{extra_context}"
                )
                with st.spinner("Re-refinando prompt..."):
                    try:
                        refined, changes = refine_prompt(client, old_prompt, combined_analysis)
                        st.session_state.refiner_result = refined
                        st.session_state.refiner_changes = changes
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao re-refinar: {e}")

        st.divider()
        if st.button("Limpar resultado", use_container_width=True, key="btn_clear_refiner"):
            st.session_state.refiner_result = None
            st.session_state.refiner_changes = None
            st.session_state.refiner_analysis = None
            st.rerun()
