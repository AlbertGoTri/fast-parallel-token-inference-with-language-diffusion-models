import json
import os


def flatten_components(components):
    """Recursively flatten nested componentResults from promptfoo's grouped assertions.
    
    Promptfoo groups llm-rubric assertions by provider into a single batch call,
    storing results as nested componentResults inside componentResults. This function
    flattens them so each individual assertion result is counted separately.
    """
    flat = []
    for comp in components:
        inner = comp.get('componentResults')
        if inner:
            flat.extend(flatten_components(inner))
        else:
            flat.append(comp)
    return flat


def generate_report(json_path="promptfoo_results.json", output_path="evaluation_report.html"):
    if not os.path.exists(json_path):
        print(f"ERROR: No se encuentra '{json_path}'.")
        print("Asegúrate de ejecutar primero: npx promptfoo eval -o promptfoo_results.json")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    raw_results = data.get('results', [])
    
    # Handle nested structure of newer promptfoo versions where data['results'] is a dict
    if isinstance(raw_results, dict) and 'results' in raw_results:
        results = raw_results['results']
    else:
        results = raw_results
    
    # Handle both old array format and newer nested results format
    if isinstance(results, list) and len(results) > 0 and isinstance(results[0], str):
        # Fallback if structure is extremely old or unexpected
        print("Warning: Unrecognized simple string array results.")
        return
        
    if isinstance(data, list):
        results = data
        
    total_prompts = len(results)
    
    # Calcular estadísticas globales (basadas en los asertos individuales, no el macro)
    total_assertions = 0
    passed_assertions = 0
    
    prompts_html = ""

    for i, res in enumerate(results):
        prompt_text = res.get('prompt', {}).get('raw', 'Sin prompt')
        response_obj = res.get('response') or {}
        model_output = response_obj.get('output', res.get('error', 'Error en la respuesta'))
        grading = res.get('gradingResult') or {}
        raw_components = grading.get('componentResults', [])
        components = flatten_components(raw_components)

        if not components and res.get('error'):
            components = [{"pass": False, "score": 0, "reason": res.get('error')}]

        # Componentes / Asertos
        assertions_html = ""
        prompt_passed_asserts = 0
        
        for comp in components:
            total_assertions += 1
            is_pass = comp.get('pass', False)
            if is_pass:
                passed_assertions += 1
                prompt_passed_asserts += 1
                
            status_color = "bg-emerald-500/20 text-emerald-400 border-emerald-500/50" if is_pass else "bg-rose-500/20 text-rose-400 border-rose-500/50"
            status_icon = "✓ PASS" if is_pass else "✗ FAIL"
            reason = comp.get('reason', 'Sin justificación o error subyacente')
            # Extract rubric question from the assertion value (Python code block)
            # The question is in: return judge(output, "...question...") or return judge(output, '...question...')
            assertion_value_raw = comp.get('assertion', {}).get('value', reason)
            # Try to extract the question string from the judge() call
            import re
            question_match = re.search(r'return judge\(output,\s*"((?:[^"\\]|\\.)*)"\s*\)', assertion_value_raw)
            if not question_match:
                question_match = re.search(r"return judge\(output,\s*'((?:[^'\\]|\\.)*)'\s*\)", assertion_value_raw)
            if question_match:
                assertion_value = question_match.group(1).strip()
            else:
                # Fallback: use raw value but truncate if too long
                assertion_value = assertion_value_raw[:200] + "..." if len(assertion_value_raw) > 200 else assertion_value_raw
            
            assertions_html += f"""
            <div class="mb-3 p-4 rounded-xl border border-white/5 bg-white/5 hover:bg-white/10 transition duration-300">
                <div class="flex items-start justify-between">
                    <div class="text-sm font-medium text-gray-300 w-3/4">{assertion_value}</div>
                    <div class="px-3 py-1 rounded-full text-xs font-bold border {status_color} backdrop-blur-sm shadow-lg">
                        {status_icon}
                    </div>
                </div>
            </div>
            """

        prompt_score = (prompt_passed_asserts / len(components) * 100) if components else 0
        prompt_macro_color = "border-emerald-500/30" if prompt_score >= 80 else "border-amber-500/30" if prompt_score >= 50 else "border-rose-500/30"

        prompts_html += f"""
        <div class="mb-8 p-6 rounded-2xl border {prompt_macro_color} bg-black/40 backdrop-blur-md shadow-2xl relative overflow-hidden group">
            <div class="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div class="flex justify-between items-center mb-4">
                <h2 class="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-400">Prompt #{i+1}</h2>
                <div class="text-lg font-mono font-bold text-gray-300">{prompt_score:.0f}% <span class="text-xs text-gray-500 font-sans">({prompt_passed_asserts}/{len(components)})</span></div>
            </div>
            <div class="mb-4">
                <h3 class="text-xs uppercase tracking-widest text-gray-500 mb-2 font-semibold">User Input</h3>
                <div class="p-4 rounded-xl bg-gray-900/80 text-gray-300 font-mono text-sm border border-gray-800 shadow-inner">
                    {prompt_text}
                </div>
            </div>
            <div class="mb-6">
                <h3 class="text-xs uppercase tracking-widest text-gray-500 mb-2 font-semibold">LLaDA Output</h3>
                <div class="p-4 rounded-xl bg-blue-950/20 text-blue-100 font-sans text-sm border border-blue-900/30 shadow-inner whitespace-pre-wrap max-h-60 overflow-y-auto custom-scrollbar">
                    {model_output}
                </div>
            </div>
            <div>
                <h3 class="text-xs uppercase tracking-widest text-gray-500 mb-3 font-semibold">Yes/No Evaluation Rubric</h3>
                <div class="pl-2 border-l-2 border-white/10">
                    {assertions_html}
                </div>
            </div>
        </div>
        """

    overall_accuracy = (passed_assertions / total_assertions * 100) if total_assertions > 0 else 0
    accuracy_color = "text-emerald-400" if overall_accuracy >= 80 else "text-amber-400" if overall_accuracy >= 50 else "text-rose-400"

    # Sanity check: warn if assertion count doesn't match expected
    if total_assertions != 60:
        print(f"WARNING: Expected 60 assertions, found {total_assertions}. JSON structure may have changed.")
    else:
        print(f"[+] Assertion count check: {total_assertions}/60 ✓")

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LLaDA Evaluation Report 2026</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
            body {{
                font-family: 'Inter', sans-serif;
                background-color: #050505;
                background-image: 
                    radial-gradient(circle at 15% 50%, rgba(30, 58, 138, 0.15), transparent 25%),
                    radial-gradient(circle at 85% 30%, rgba(88, 28, 135, 0.15), transparent 25%);
                background-attachment: fixed;
                color: #e5e5e5;
            }}
            code, font-mono {{
                font-family: 'JetBrains Mono', monospace;
            }}
            .custom-scrollbar::-webkit-scrollbar {{
                width: 6px;
            }}
            .custom-scrollbar::-webkit-scrollbar-track {{
                background: rgba(255, 255, 255, 0.02);
                border-radius: 4px;
            }}
            .custom-scrollbar::-webkit-scrollbar-thumb {{
                background: rgba(255, 255, 255, 0.1);
                border-radius: 4px;
            }}
            .glass-header {{
                background: rgba(10, 10, 10, 0.7);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            }}
        </style>
    </head>
    <body class="antialiased min-h-screen flex flex-col">
        
        <header class="glass-header sticky top-0 z-50 px-6 py-4 shadow-2xl">
            <div class="max-w-6xl mx-auto flex justify-between items-center">
                <div class="flex items-center gap-3">
                    <div class="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-500 to-purple-600 animate-pulse flex items-center justify-center shadow-lg shadow-blue-500/20">
                        <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                    </div>
                    <h1 class="text-2xl font-bold tracking-tight">LLaDA <span class="font-light text-gray-400">Eval Pipeline</span></h1>
                </div>
                <div class="text-right">
                    <p class="text-xs uppercase tracking-wider text-gray-500 font-semibold mb-1">Global Accuracy</p>
                    <p class="text-3xl font-mono font-bold {accuracy_color} drop-shadow-md">
                        {overall_accuracy:.1f}%
                    </p>
                </div>
            </div>
        </header>

        <main class="flex-grow p-6 py-10 relative z-10">
            <div class="max-w-6xl mx-auto">
                <div class="mb-10 p-6 rounded-2xl bg-gradient-to-b from-white/5 to-transparent border border-white/10 backdrop-blur-[2px]">
                    <div class="grid grid-cols-3 gap-6 text-center divide-x divide-white/10">
                        <div>
                            <p class="text-sm text-gray-500 mb-1">Total Prompts</p>
                            <p class="text-2xl font-semibold text-white">{total_prompts}</p>
                        </div>
                        <div>
                            <p class="text-sm text-gray-500 mb-1">Total Assertions</p>
                            <p class="text-2xl font-semibold text-white">{total_assertions}</p>
                        </div>
                        <div>
                            <p class="text-sm text-gray-500 mb-1">Passed</p>
                            <p class="text-2xl font-semibold text-emerald-400">{passed_assertions}</p>
                        </div>
                    </div>
                </div>

                <div class="space-y-6">
                    {prompts_html}
                </div>
            </div>
        </main>

        <footer class="mt-auto py-6 border-t border-white/5 text-center text-xs text-gray-600">
            Generated with Promptfoo Context &bull; AI Evaluation Pipeline 2026
        </footer>
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[+] ¡Reporte HTML futurista generado con éxito en '{output_path}'!")
    print(f"    Precisión global calculada: {overall_accuracy:.1f}%")

if __name__ == "__main__":
    generate_report()