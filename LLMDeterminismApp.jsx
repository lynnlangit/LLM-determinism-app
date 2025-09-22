const LLMDeterminismApp = () => {
  const [activeTab, setActiveTab] = React.useState('overview');
  const [isDeterministic, setIsDeterministic] = React.useState(false);
  const [floatA, setFloatA] = React.useState(0.1);
  const [floatB, setFloatB] = React.useState(0.2);
  const [floatC, setFloatC] = React.useState(0.3);
  const [batchSize, setBatchSize] = React.useState(2);
  const [kernelType, setKernelType] = React.useState('RMSNorm');
  const [demoRuns, setDemoRuns] = React.useState([]);
  const [isRunningDemo, setIsRunningDemo] = React.useState(false);

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'float', label: 'Float' },
    { id: 'batch', label: 'Batch' },
    { id: 'atomic', label: 'Atomic' },
    { id: 'kernels', label: 'Kernels' },
    { id: 'demo', label: 'Demo' },
    { id: 'performance', label: 'Performance' },
    { id: 'solution', label: 'Solution' }
  ];

  const styles = {
    container: {
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)',
      fontFamily: 'Arial, sans-serif',
      color: 'white',
      padding: '20px'
    },
    main: {
      maxWidth: '1200px',
      margin: '0 auto'
    },
    header: {
      textAlign: 'center',
      marginBottom: '30px'
    },
    title: {
      fontSize: '2.5rem',
      fontWeight: 'bold',
      marginBottom: '10px',
      textShadow: '2px 2px 4px rgba(0,0,0,0.3)'
    },
    subtitle: {
      fontSize: '1.2rem',
      opacity: 0.9,
      marginBottom: '20px'
    },
    modeToggle: {
      background: isDeterministic ? '#10b981' : '#ef4444',
      border: 'none',
      padding: '12px 24px',
      borderRadius: '25px',
      color: 'white',
      fontSize: '16px',
      fontWeight: 'bold',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      boxShadow: '0 4px 15px rgba(0,0,0,0.2)'
    },
    tabContainer: {
      display: 'flex',
      flexWrap: 'wrap',
      justifyContent: 'center',
      marginBottom: '30px',
      gap: '10px'
    },
    tab: {
      background: 'rgba(255,255,255,0.1)',
      border: '1px solid rgba(255,255,255,0.2)',
      padding: '12px 20px',
      borderRadius: '8px',
      color: 'white',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      fontSize: '14px',
      fontWeight: '500'
    },
    activeTab: {
      background: '#3b82f6',
      borderColor: '#3b82f6',
      transform: 'translateY(-2px)',
      boxShadow: '0 4px 15px rgba(59, 130, 246, 0.3)'
    },
    content: {
      background: 'rgba(255,255,255,0.05)',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: '12px',
      padding: '30px',
      minHeight: '600px'
    },
    section: {
      background: 'rgba(255,255,255,0.05)',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: '8px',
      padding: '20px',
      marginBottom: '20px'
    },
    input: {
      background: 'rgba(255,255,255,0.1)',
      border: '1px solid rgba(255,255,255,0.3)',
      borderRadius: '6px',
      padding: '10px',
      color: 'white',
      fontSize: '14px',
      width: '100px'
    },
    button: {
      background: '#3b82f6',
      border: 'none',
      padding: '10px 20px',
      borderRadius: '6px',
      color: 'white',
      cursor: 'pointer',
      fontSize: '14px',
      fontWeight: '500',
      transition: 'all 0.3s ease'
    },
    code: {
      background: 'rgba(0,0,0,0.3)',
      border: '1px solid rgba(255,255,255,0.2)',
      borderRadius: '6px',
      padding: '15px',
      fontFamily: 'Monaco, Consolas, monospace',
      fontSize: '13px',
      lineHeight: '1.4',
      overflowX: 'auto',
      marginBottom: '15px'
    },
    grid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
      gap: '20px',
      marginBottom: '20px'
    },
    warning: {
      background: 'rgba(251, 191, 36, 0.1)',
      border: '1px solid #fbbf24',
      borderRadius: '6px',
      padding: '15px',
      color: '#fbbf24'
    },
    success: {
      background: 'rgba(16, 185, 129, 0.1)',
      border: '1px solid #10b981',
      borderRadius: '6px',
      padding: '15px',
      color: '#10b981'
    },
    error: {
      background: 'rgba(239, 68, 68, 0.1)',
      border: '1px solid #ef4444',
      borderRadius: '6px',
      padding: '15px',
      color: '#ef4444'
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(() => {
      console.log('Copied to clipboard');
    });
  };

  const calculateFloat = () => {
    const result1 = (floatA + floatB) - floatC;
    const result2 = floatA + (floatB - floatC);
    return { result1, result2, differs: Math.abs(result1 - result2) > 1e-15 };
  };

  const getBatchOutput = (size, deterministic) => {
    if (deterministic) return "Queens, New York";
    const outputs = ["Queens, New York", "New York City"];
    return size >= 8 ? outputs[Math.floor(Math.random() * outputs.length)] : outputs[0];
  };

  const getAtomicResults = (deterministic) => {
    const values = [1.5, 2.3, 3.7, 4.2];
    if (deterministic) {
      return { sum: 11.7, order: "Tree reduction (fixed order)" };
    }
    const shuffled = [...values].sort(() => Math.random() - 0.5);
    return { sum: 11.7 + (Math.random() - 0.5) * 0.001, order: `Atomic adds: ${shuffled.join(' + ')}` };
  };

  const runGemmaDemo = async () => {
    setIsRunningDemo(true);
    setDemoRuns([]);

    for (let i = 0; i < 3; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      const run = {
        id: i + 1,
        batchSize: [1, 4, 8][i],
        output: isDeterministic ?
          "Quantum computing uses quantum mechanics principles like superposition and entanglement to process information exponentially faster than classical computers." :
          [
            "Quantum computing uses quantum mechanics principles like superposition and entanglement to process information exponentially faster than classical computers.",
            "Quantum computing leverages quantum mechanical phenomena to perform calculations that would be impossible for traditional computers.",
            "Quantum computers utilize quantum bits (qubits) that can exist in multiple states simultaneously, enabling parallel processing of complex problems."
          ][i],
        logprob: isDeterministic ? -2.341 : -2.341 + (Math.random() - 0.5) * 0.1
      };
      setDemoRuns(prev => [...prev, run]);
    }
    setIsRunningDemo(false);
  };

  const kernelStrategies = {
    RMSNorm: {
      description: "Root Mean Square Layer Normalization",
      strategy: "Use deterministic reduction algorithms and avoid batch-dependent optimizations",
      code: `def rms_norm_deterministic(x):
    # Force single-batch processing
    variance = torch.mean(x * x, dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps)`
    },
    "Matrix Multiplication": {
      description: "Large matrix operations using GEMM",
      strategy: "Disable TensorFloat-32 and use consistent CUDA algorithms",
      code: `torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)`
    },
    Attention: {
      description: "Self-attention mechanism",
      strategy: "Use flash attention with deterministic mode and fixed sequence processing",
      code: `def attention_deterministic(q, k, v):
    # Disable flash attention optimizations
    return F.scaled_dot_product_attention(
        q, k, v, is_causal=True,
        enable_flash=False
    )`
    }
  };

  const renderOverview = () => (
    <div>
      <h2 style={{fontSize: '2rem', marginBottom: '20px'}}>The Nondeterminism Problem</h2>

      <div style={styles.section}>
        <h3 style={{color: '#fbbf24'}}>Qwen-235B Experiment Results</h3>
        <p>When researchers ran the same prompt 1000 times with temperature=0:</p>
        <ul style={{lineHeight: '1.6'}}>
          <li><strong>80 unique outputs</strong> were generated</li>
          <li><strong>Token 103 divergence:</strong> 992 runs produced "Queens, New York" vs 8 runs "New York City"</li>
          <li>This proves that temperature=0 doesn't guarantee determinism</li>
        </ul>
      </div>

      <div style={styles.section}>
        <h3 style={{color: '#ef4444'}}>Root Causes of Nondeterminism</h3>
        <div style={styles.grid}>
          <div>
            <h4>1. Floating-Point Operations</h4>
            <p>Non-associative arithmetic in GPU computations</p>
          </div>
          <div>
            <h4>2. Batch Size Variance</h4>
            <p>Different kernel optimizations for different batch sizes</p>
          </div>
          <div>
            <h4>3. Concurrent Execution</h4>
            <p>Race conditions in parallel GPU operations</p>
          </div>
        </div>
      </div>

      <div style={isDeterministic ? styles.success : styles.warning}>
        <strong>Current Mode: {isDeterministic ? 'Deterministic' : 'Non-deterministic'}</strong>
        <p>This mode affects all demonstrations in the app.</p>
      </div>
    </div>
  );

  const renderFloat = () => {
    const calc = calculateFloat();
    return (
      <div>
        <h2 style={{fontSize: '2rem', marginBottom: '20px'}}>Floating-Point Non-Associativity</h2>

        <div style={styles.section}>
          <h3>Interactive Calculator</h3>
          <p>Demonstrate how (a+b)-c ≠ a+(b-c) due to floating-point precision:</p>

          <div style={{display: 'flex', gap: '20px', alignItems: 'center', marginBottom: '20px'}}>
            <div>
              <label>a: </label>
              <input
                type="number"
                step="0.001"
                value={floatA}
                onChange={(e) => setFloatA(parseFloat(e.target.value))}
                style={styles.input}
              />
            </div>
            <div>
              <label>b: </label>
              <input
                type="number"
                step="0.001"
                value={floatB}
                onChange={(e) => setFloatB(parseFloat(e.target.value))}
                style={styles.input}
              />
            </div>
            <div>
              <label>c: </label>
              <input
                type="number"
                step="0.001"
                value={floatC}
                onChange={(e) => setFloatC(parseFloat(e.target.value))}
                style={styles.input}
              />
            </div>
            <button
              style={{...styles.button, fontSize: '18px'}}
              onClick={() => {
                setFloatA(Math.random());
                setFloatB(Math.random());
                setFloatC(Math.random());
              }}
            >
              🔄
            </button>
          </div>

          <div style={styles.code}>
            <div>(a + b) - c = ({floatA} + {floatB}) - {floatC} = <span style={{color: '#10b981'}}>{calc.result1}</span></div>
            <div>a + (b - c) = {floatA} + ({floatB} - {floatC}) = <span style={{color: '#10b981'}}>{calc.result2}</span></div>
            <div>Difference: {Math.abs(calc.result1 - calc.result2).toExponential()}</div>
          </div>

          {calc.differs && (
            <div style={styles.warning}>
              ⚠️ Results differ! This demonstrates floating-point non-associativity.
            </div>
          )}
        </div>

        <div style={styles.section}>
          <h3>GPU Implications</h3>
          <p>In GPU operations, the order of floating-point additions can vary based on:</p>
          <ul>
            <li>Thread scheduling</li>
            <li>Memory access patterns</li>
            <li>Kernel optimization choices</li>
          </ul>
        </div>
      </div>
    );
  };

  const renderBatch = () => (
    <div>
      <h2 style={{fontSize: '2rem', marginBottom: '20px'}}>Batch Size Impact</h2>

      <div style={styles.section}>
        <h3>Kernel Selection Simulation</h3>
        <p>Different batch sizes trigger different optimized kernels:</p>

        <div style={{marginBottom: '20px'}}>
          <label>Batch Size: </label>
          <select
            value={batchSize}
            onChange={(e) => setBatchSize(parseInt(e.target.value))}
            style={styles.input}
          >
            <option value={2}>2</option>
            <option value={4}>4</option>
            <option value={8}>8</option>
            <option value={16}>16</option>
          </select>
        </div>

        <div style={styles.code}>
          <div>Batch Size: {batchSize}</div>
          <div>Kernel: {batchSize >= 8 ? 'cuBLAS (optimized)' : 'Custom kernel'}</div>
          <div>Output: "{getBatchOutput(batchSize, isDeterministic)}"</div>
        </div>

        <div style={styles.grid}>
          <div style={styles.section}>
            <h4>Small Batches (2-4)</h4>
            <p>Use custom kernels with predictable behavior</p>
          </div>
          <div style={styles.section}>
            <h4>Large Batches (8+)</h4>
            <p>Trigger cuBLAS optimizations that may vary</p>
          </div>
        </div>
      </div>
    </div>
  );

  const renderAtomic = () => {
    const results = getAtomicResults(isDeterministic);
    return (
      <div>
        <h2 style={{fontSize: '2rem', marginBottom: '20px'}}>Atomic Operations & Race Conditions</h2>

        <div style={styles.section}>
          <h3>GPU Addition Simulation</h3>
          <p>Adding values [1.5, 2.3, 3.7, 4.2] using different strategies:</p>

          <div style={styles.code}>
            <div>Values: [1.5, 2.3, 3.7, 4.2]</div>
            <div>Strategy: {results.order}</div>
            <div>Result: {results.sum.toFixed(6)}</div>
          </div>

          <button
            style={styles.button}
            onClick={() => setIsDeterministic(!isDeterministic)}
          >
            Toggle Strategy
          </button>
        </div>

        <div style={styles.grid}>
          <div style={styles.section}>
            <h4 style={{color: '#ef4444'}}>Atomic Adds (Non-deterministic)</h4>
            <p>Multiple threads add values simultaneously, order depends on thread scheduling</p>
          </div>
          <div style={styles.section}>
            <h4 style={{color: '#10b981'}}>Tree Reduction (Deterministic)</h4>
            <p>Fixed hierarchical reduction ensures consistent order</p>
          </div>
        </div>
      </div>
    );
  };

  const renderKernels = () => {
    const strategy = kernelStrategies[kernelType];
    return (
      <div>
        <h2 style={{fontSize: '2rem', marginBottom: '20px'}}>Kernel-Level Solutions</h2>

        <div style={styles.section}>
          <h3>Select Operation Type</h3>
          <select
            value={kernelType}
            onChange={(e) => setKernelType(e.target.value)}
            style={{...styles.input, width: '200px'}}
          >
            {Object.keys(kernelStrategies).map(key => (
              <option key={key} value={key}>{key}</option>
            ))}
          </select>
        </div>

        <div style={styles.section}>
          <h3>{strategy.description}</h3>
          <p><strong>Strategy:</strong> {strategy.strategy}</p>

          <h4>Implementation:</h4>
          <div style={styles.code}>
            {strategy.code}
          </div>

          <button
            style={styles.button}
            onClick={() => copyToClipboard(strategy.code)}
          >
            Copy Code
          </button>
        </div>
      </div>
    );
  };

  const renderDemo = () => (
    <div>
      <h2 style={{fontSize: '2rem', marginBottom: '20px'}}>Gemma-2B Live Simulation</h2>

      <div style={styles.section}>
        <h3>Prompt: "Explain quantum computing in one sentence."</h3>

        <button
          style={{...styles.button, marginBottom: '20px'}}
          onClick={runGemmaDemo}
          disabled={isRunningDemo}
        >
          {isRunningDemo ? 'Running...' : 'Run 3 Passes'}
        </button>

        {demoRuns.map(run => (
          <div key={run.id} style={styles.section}>
            <h4>Run {run.id} (Batch Size: {run.batchSize})</h4>
            <div style={styles.code}>
              <div>Output: "{run.output}"</div>
              <div>Log Probability: {run.logprob.toFixed(3)}</div>
            </div>
          </div>
        ))}
      </div>

      <div style={styles.section}>
        <h3>Code Examples</h3>
        <div style={styles.grid}>
          <div>
            <h4>Standard Implementation</h4>
            <div style={styles.code}>
{`model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    torch_dtype=torch.float16
)
output = model.generate(
    inputs,
    temperature=0.0,
    max_new_tokens=50
)`}
            </div>
            <button style={styles.button} onClick={() => copyToClipboard(`model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    torch_dtype=torch.float16
)
output = model.generate(
    inputs,
    temperature=0.0,
    max_new_tokens=50
)`)}>Copy</button>
          </div>

          <div>
            <h4>Deterministic Implementation</h4>
            <div style={styles.code}>
{`torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    torch_dtype=torch.float32
)
output = model.generate(
    inputs,
    temperature=0.0,
    do_sample=False,
    batch_size=1
)`}
            </div>
            <button style={styles.button} onClick={() => copyToClipboard(`torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    torch_dtype=torch.float32
)
output = model.generate(
    inputs,
    temperature=0.0,
    do_sample=False,
    batch_size=1
)`)}>Copy</button>
          </div>

          <div>
            <h4>vLLM Deterministic</h4>
            <div style={styles.code}>
{`from vllm import LLM

llm = LLM(
    model="google/gemma-2b",
    enforce_eager=True,
    dtype="float32"
)
output = llm.generate(
    prompts,
    temperature=0.0,
    seed=42
)`}
            </div>
            <button style={styles.button} onClick={() => copyToClipboard(`from vllm import LLM

llm = LLM(
    model="google/gemma-2b",
    enforce_eager=True,
    dtype="float32"
)
output = llm.generate(
    prompts,
    temperature=0.0,
    seed=42
)`)}>Copy</button>
          </div>
        </div>
      </div>
    </div>
  );

  const renderPerformance = () => (
    <div>
      <h2 style={{fontSize: '2rem', marginBottom: '20px'}}>Performance Impact</h2>

      <div style={styles.section}>
        <h3>Throughput Comparison</h3>

        <div style={{marginBottom: '30px'}}>
          <div style={{marginBottom: '15px'}}>
            <div>Standard Gemma-2B: 50 tokens/second</div>
            <div style={{
              background: '#3b82f6',
              height: '30px',
              width: '100%',
              borderRadius: '4px',
              display: 'flex',
              alignItems: 'center',
              paddingLeft: '10px',
              marginTop: '5px'
            }}>
              100%
            </div>
          </div>

          <div style={{marginBottom: '15px'}}>
            <div>Deterministic Gemma-2B: 35 tokens/second</div>
            <div style={{
              background: '#fbbf24',
              height: '30px',
              width: '70%',
              borderRadius: '4px',
              display: 'flex',
              alignItems: 'center',
              paddingLeft: '10px',
              marginTop: '5px'
            }}>
              70%
            </div>
          </div>
        </div>
      </div>

      <div style={styles.section}>
        <h3>Real-World Results (vLLM)</h3>
        <div style={styles.grid}>
          <div>
            <h4>Before Deterministic Mode</h4>
            <div style={styles.code}>
              Time: 26 seconds<br/>
              Throughput: 38.5 tok/s
            </div>
          </div>
          <div>
            <h4>After Deterministic Mode</h4>
            <div style={styles.code}>
              Time: 42 seconds<br/>
              Throughput: 23.8 tok/s<br/>
              <span style={{color: '#fbbf24'}}>1.6x slower</span>
            </div>
          </div>
        </div>
      </div>

      <div style={styles.section}>
        <h3>Trade-offs</h3>
        <div style={styles.grid}>
          <div style={styles.success}>
            <h4>Benefits</h4>
            <ul>
              <li>Reproducible results</li>
              <li>Easier debugging</li>
              <li>Scientific reproducibility</li>
            </ul>
          </div>
          <div style={styles.warning}>
            <h4>Costs</h4>
            <ul>
              <li>30-40% throughput reduction</li>
              <li>Higher memory usage</li>
              <li>Limited optimization</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );

  const renderSolution = () => (
    <div>
      <h2 style={{fontSize: '2rem', marginBottom: '20px'}}>Implementation Guide</h2>

      <div style={styles.section}>
        <h3>Step 1: Environment Setup</h3>
        <div style={styles.code}>
{`# Set deterministic algorithms
export PYTORCH_CUDA_ALLOC_CONF=deterministic_algorithms=true
export CUBLAS_WORKSPACE_CONFIG=:16:8

# Install requirements
pip install torch transformers vllm`}
        </div>
      </div>

      <div style={styles.section}>
        <h3>Step 2: Code Configuration</h3>
        <div style={styles.code}>
{`import torch
import random
import numpy as np

# Set all random seeds
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Enable deterministic algorithms
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False`}
        </div>
      </div>

      <div style={styles.section}>
        <h3>Step 3: Model Loading</h3>
        <div style={styles.code}>
{`from transformers import AutoModelForCausalLM, AutoTokenizer

# Use float32 instead of float16 for determinism
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    torch_dtype=torch.float32,  # Important!
    device_map="auto"
)`}
        </div>
      </div>

      <div style={styles.section}>
        <h3>Step 4: Generation Parameters</h3>
        <div style={styles.code}>
{`output = model.generate(
    input_ids,
    temperature=0.0,
    do_sample=False,        # Disable sampling
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id,
    batch_size=1           # Force single batch
)`}
        </div>
      </div>

      <div style={styles.section}>
        <h3>Use Cases & Trade-offs</h3>
        <div style={styles.grid}>
          <div>
            <h4 style={{color: '#10b981'}}>When to Use Deterministic</h4>
            <ul>
              <li>Research experiments</li>
              <li>A/B testing</li>
              <li>Debugging model behavior</li>
              <li>Compliance requirements</li>
            </ul>
          </div>
          <div>
            <h4 style={{color: '#ef4444'}}>When to Avoid</h4>
            <ul>
              <li>Production serving</li>
              <li>High-throughput applications</li>
              <li>Real-time inference</li>
              <li>Resource-constrained environments</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );

  const renderContent = () => {
    switch(activeTab) {
      case 'overview': return renderOverview();
      case 'float': return renderFloat();
      case 'batch': return renderBatch();
      case 'atomic': return renderAtomic();
      case 'kernels': return renderKernels();
      case 'demo': return renderDemo();
      case 'performance': return renderPerformance();
      case 'solution': return renderSolution();
      default: return renderOverview();
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.main}>
        <header style={styles.header}>
          <h1 style={styles.title}>Defeating Nondeterminism in LLM Inference</h1>
          <p style={styles.subtitle}>
            Understanding and controlling randomness in Gemma and other language models
          </p>
          <button
            style={styles.modeToggle}
            onClick={() => setIsDeterministic(!isDeterministic)}
            onMouseEnter={(e) => e.target.style.transform = 'scale(1.05)'}
            onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
          >
            {isDeterministic ? '✓ Deterministic Mode' : '⚠ Non-deterministic Mode'}
          </button>
        </header>

        <nav style={styles.tabContainer}>
          {tabs.map(tab => (
            <button
              key={tab.id}
              style={{
                ...styles.tab,
                ...(activeTab === tab.id ? styles.activeTab : {})
              }}
              onClick={() => setActiveTab(tab.id)}
              onMouseEnter={(e) => {
                if (activeTab !== tab.id) {
                  e.target.style.background = 'rgba(255,255,255,0.2)';
                }
              }}
              onMouseLeave={(e) => {
                if (activeTab !== tab.id) {
                  e.target.style.background = 'rgba(255,255,255,0.1)';
                }
              }}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        <main style={styles.content}>
          {renderContent()}
        </main>
      </div>
    </div>
  );
};

// Export for browser use
window.LLMDeterminismApp = LLMDeterminismApp;