"""
Resume Skill Analyzer — 10-Step Pipeline  v17
Hybrid extraction: structured section parser + NLP for prose
Offline mode: TF-IDF cosine similarity (no HuggingFace / internet required)
"""
from docx import Document
import re, fitz, spacy
import numpy as np
from collections import defaultdict
from flask import Flask, render_template, request, session, redirect, url_for, flash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document

app = Flask(__name__)
app.secret_key = "supersecretkey"
nlp  = spacy.load("en_core_web_sm")
ALLOWED = {"pdf", "docx", "txt"}
APP_VERSION = "v17"

# ══════════════════════════════════════════════════════════════════════════════
# SKILL ONTOLOGY
# ══════════════════════════════════════════════════════════════════════════════
SKILL_ONTOLOGY = {
    "Programming Languages": [
        "python","java","javascript","typescript","c++","c#","scala","golang",
        "rust","kotlin","swift","php","ruby","r","matlab","bash","perl","dart","c",
        "haskell","erlang","elixir","clojure","f#","ocaml","lua","groovy","julia",
        "cobol","fortran","assembly","vba","powershell","awk","sed","tcl","prolog",
        "lisp","scheme","smalltalk","ada","pascal","d","nim","zig","v","crystal",
        "objective-c","abap","apex","solidity","move","cairo",
    ],
    "Web Development": [
        "react","angular","vue","node.js","django","flask","fastapi","spring",
        "html","css","graphql","rest api","next.js","tailwind","bootstrap",
        "svelte","nuxt.js","gatsby","remix","astro","solid.js","qwik",
        "jquery","webpack","vite","rollup","parcel","babel","esbuild",
        "express.js","nestjs","koa","hapi","laravel","symfony","rails","asp.net",
        "spring boot","quarkus","micronaut","grpc","websocket","oauth","jwt",
        "http","https","web sockets","service workers","pwa","ssr","spa","ssg",
        "sass","less","styled components","css modules","material ui","ant design",
        "chakra ui","shadcn","radix ui","storybook","cypress","playwright","jest",
        "vitest","testing library","selenium","puppeteer","api design","openapi",
        "swagger","xml","json","yaml","toml",
    ],
    "Data Engineering": [
        "etl","elt","data pipeline","apache spark","kafka","airflow","dbt",
        "hadoop","hive","data warehouse","data modeling","data transformation",
        "data validation","pyspark","databricks","redshift","bigquery","snowflake",
        "apache flink","apache beam","apache nifi","luigi","prefect","dagster",
        "mage","glue","data lake","data lakehouse","delta lake","iceberg","hudi",
        "druid","presto","trino","dask","ray","polars","arrow","parquet","avro",
        "orc","data quality","data lineage","data catalog","data governance",
        "data mesh","data fabric","change data capture","cdc","stream processing",
        "batch processing","real-time processing","azure data factory",
        "azure synapse","aws glue","gcp dataflow","gcp dataproc","fivetran","airbyte",
        "stitch","talend","informatica","ssis","ssas","ssrs","power query",
        "data profiling","data enrichment","data cleansing","master data management",
    ],
    "Data Science & ML": [
        "machine learning","deep learning","neural network","neural networks",
        "natural language processing","computer vision","tensorflow","pytorch",
        "keras","hugging face","transformers","bert","gpt","llm","diffusion model",
        "reinforcement learning","transfer learning","fine-tuning",
        "convolutional neural network","cnn","recurrent neural network","rnn",
        "lstm","gru","generative ai","gan","stable diffusion","langchain",
        "scikit-learn","pandas","numpy","data analysis","statistical analysis",
        "feature engineering","xgboost","lightgbm","catboost","scipy","seaborn",
        "matplotlib","statsmodels","regression","classification","clustering",
        "data visualization","hypothesis testing","time series","data cleaning",
        "data wrangling","jupyter","jupyter notebook","colab","kaggle",
        "mlflow","weights and biases","onnx","tensorrt","cuda","cudnn",
        "llama","mistral","falcon","openai api","anthropic api","gemini",
        "langraph","llamaindex","vector database","rag","retrieval augmented generation",
        "embeddings","semantic search","pinecone","weaviate","chroma","qdrant",
        "faiss","ann","approximate nearest neighbor","prompt engineering",
        "few-shot learning","zero-shot learning","meta-learning",
        "federated learning","active learning","self-supervised learning",
        "contrastive learning","knowledge distillation","model compression",
        "quantization","pruning","hyperparameter tuning","optuna","ray tune",
        "automl","tpot","h2o","pycaret","a/b testing","causal inference",
        "bayesian inference","mcmc","variational inference","survival analysis",
        "anomaly detection","recommendation system","collaborative filtering",
        "matrix factorization","object detection","image segmentation","yolo",
        "resnet","vgg","efficientnet","vision transformer","vit","clip","dall-e",
        "whisper","speech recognition","text to speech","ocr","plotly","bokeh",
        "altair","dash","streamlit","gradio","tensorboard","wandb",
        "feature store","model registry","model monitoring","data drift",
        "model explainability","shap","lime","interpretable ml",
    ],
    "Cloud & DevOps": [
        "aws","azure","gcp","docker","kubernetes","linux","ci/cd",
        "terraform","ansible","jenkins","devops","helm","nginx",
        "aws lambda","aws ec2","aws s3","aws rds","aws sagemaker","aws ecs",
        "aws eks","aws cloudformation","aws cdk","aws cloudwatch",
        "azure devops","azure kubernetes service","azure functions","azure blob",
        "azure active directory","azure monitor","gcp cloud run","gcp pubsub",
        "gcp gke","gcp cloud functions","gcp vertex ai","gcp cloud storage",
        "prometheus","grafana","datadog","new relic","splunk","elk stack",
        "elasticsearch","logstash","kibana","fluentd","istio","envoy","consul",
        "vault","packer","vagrant","podman","containerd","openshift",
        "argocd","flux","gitops","github actions","gitlab ci","circle ci",
        "travis ci","azure pipelines","bitbucket pipelines","tekton",
        "infrastructure as code","site reliability engineering","sre",
        "observability","monitoring","logging","tracing","opentelemetry",
        "chaos engineering","load balancing","auto scaling","serverless",
        "microservices","service mesh","api gateway","cdn","cloudflare",
        "load testing","performance testing","jmeter","k6","locust",
    ],
    "Databases": [
        "sql","mysql","postgresql","mongodb","redis","cassandra","oracle",
        "sqlite","elasticsearch","firebase","dynamodb","cosmosdb","neo4j",
        "couchdb","couchbase","influxdb","timescaledb","cockroachdb","tidb",
        "vitess","planetscale","supabase","neon","turso","airtable",
        "mariadb","ms sql server","sybase","teradata","greenplum",
        "db2","hbase","scylladb","rethinkdb","arangodb","dgraph",
        "memcached","hazelcast","apache ignite","vector database",
        "database design","database administration","dba","query optimization",
        "indexing","sharding","replication","backup and recovery",
        "database migration","orm","sqlalchemy","hibernate","prisma","drizzle",
        "stored procedures","triggers","views","transactions","acid",
        "nosql","graph database","time series database","in-memory database",
    ],
    "Embedded Systems & IoT": [
        "arduino","raspberry pi","stm32","esp32","rtos","freertos","spi","uart",
        "i2c","embedded linux","arm cortex","microcontroller","fpga","firmware",
        "sensor integration","can bus","verilog","iot","rfid","biometric","opencv",
        "embedded systems","mqtt","modbus","profibus","profinet","ethercat",
        "zigbee","z-wave","lora","lorawan","nb-iot","lte-m","bluetooth le",
        "ble","wifi","esp8266","nrf52","pic","avr","msp430","zynq",
        "system on chip","soc","vhdl","systemverilog","quartus","vivado",
        "keil","iar","segger","openocd","jtag","swd","bootloader",
        "dma","adc","dac","pwm","interrupt","watchdog","power management",
        "real-time systems","scheduling","embedded c","embedded c++",
        "device driver","bsp","board support package","uboot","yocto",
        "buildroot","zephyr","contiki","threadx","safertos",
        "functional safety","iso 26262","iec 61508","misra c",
        "pcb design","altium","kicad","eagle","schematic","bom",
    ],
    "Networking & Security": [
        "networking","tcp/ip","vpn","firewall","cybersecurity","packet tracer",
        "cisco","wireshark","penetration testing","information security",
        "network security","ethical hacking","vulnerability assessment",
        "nmap","metasploit","burp suite","kali linux","snort","suricata",
        "ips","ids","siem","soc","threat intelligence","threat hunting",
        "incident response","digital forensics","reverse engineering","malware analysis",
        "cryptography","ssl/tls","pki","zero trust","oauth2","saml","ldap",
        "active directory","identity management","iam","pam","devsecops",
        "owasp","secure coding","code review","security audit","compliance",
        "gdpr","hipaa","iso 27001","soc 2","nist","pci dss",
        "dns","dhcp","http","smtp","ftp","sftp","ssh","rdp",
        "bgp","ospf","mpls","vlan","nat","load balancer","proxy",
        "juniper","palo alto","fortinet","checkpoint","f5",
        "cloud security","container security","endpoint security","dlp",
        "network monitoring","nagios","zabbix","prtg",
    ],
    "Mobile Development": [
        "android","ios","react native","flutter","xamarin","ionic",
        "swift","swiftui","objective-c","kotlin","jetpack compose",
        "android studio","xcode","expo","capacitor","cordova",
        "mobile ui","push notifications","in-app purchase","app store",
        "google play","mobile testing","espresso","xctest","detox",
        "firebase","crashlytics","app performance","mobile security",
        "responsive design","adaptive layout","accessibility",
    ],
    "AI & Generative AI": [
        "generative ai","prompt engineering","chatgpt","gpt-4","claude",
        "llm","large language model","foundation model","multimodal",
        "ai agents","autonomous agents","tool use","function calling",
        "retrieval augmented generation","rag","vector search","embeddings",
        "langchain","langraph","llamaindex","semantic kernel","autogen",
        "crewai","openai","anthropic","cohere","mistral","ollama",
        "ai safety","alignment","responsible ai","ai ethics","bias detection",
        "model evaluation","benchmarking","red teaming","adversarial testing",
        "ai product development","ai integration","llm fine-tuning",
        "instruction tuning","rlhf","dpo","model deployment","inference optimization",
    ],
    "Teaching & Education": [
        "lesson planning","classroom management","student engagement",
        "curriculum development","online teaching","smart board","smart classroom",
        "educational software","student progress","blended learning",
        "digital tools","ms office","instructional design","e-learning",
        "lms","moodle","canvas","blackboard","google classroom",
        "assessment design","rubric design","differentiated instruction",
        "special education","inclusive education","project based learning",
        "flipped classroom","gamification","stem education","pedagogy",
        "tutoring","mentoring students","academic advising","counselling",
        "parent communication","school administration","edtech",
    ],
    "Mathematics": [
        "algebra","calculus","geometry","trigonometry","statistics",
        "linear algebra","discrete mathematics","probability","arithmetic",
        "differential equations","numerical methods","optimization",
        "real analysis","complex analysis","topology","graph theory",
        "number theory","combinatorics","abstract algebra","group theory",
        "ring theory","field theory","functional analysis","measure theory",
        "stochastic processes","markov chains","operations research",
        "linear programming","integer programming","convex optimization",
        "game theory","information theory","fourier analysis","wavelets",
        "numerical analysis","finite element method","mathematical modelling",
        "applied mathematics","pure mathematics","computational mathematics",
    ],
    "Tools & Productivity": [
        "matlab","excel","latex","tableau","power bi","google sheets",
        "ms word","ms powerpoint","jupyter notebook","google colab",
        "vs code","postman","figma","notion",
        "jira","confluence","trello","asana","monday.com","linear","clickup",
        "slack","microsoft teams","zoom","google workspace","office 365",
        "adobe xd","sketch","invision","zeplin","miro","lucidchart",
        "drawio","visio","obsidian","roam","logseq","airtable",
        "zapier","make","n8n","power automate","ifttt",
        "google analytics","mixpanel","amplitude","hotjar","clarity",
        "looker","metabase","superset","redash","grafana","kibana",
        "sentry","rollbar","pagerduty","opsgenie","statuspage",
        "github copilot","cursor","tabnine","codeium","jetbrains",
        "intellij","pycharm","webstorm","goland","datagrip","rider",
        "eclipse","netbeans","android studio","xcode","vim","neovim","emacs",
    ],
    "Version Control": [
        "git","github","gitlab","bitbucket","svn","mercurial",
        "version control","source control","git flow","trunk based development",
        "branching strategy","pull request","code review","merge conflict",
        "cherry pick","rebase","git hooks","pre-commit","semantic versioning",
    ],
    "Project Management & Agile": [
        "agile","scrum","kanban","safe","lean","waterfall","prince2","pmp",
        "sprint planning","sprint review","retrospective","daily standup",
        "product backlog","user stories","epics","story points","velocity",
        "project planning","risk management","stakeholder management",
        "resource management","budgeting","roadmap","milestones","deliverables",
        "project documentation","change management","scope management",
        "quality assurance","qa","testing","uat","release management",
        "product management","product owner","scrum master","delivery manager",
    ],
    "Design & UI/UX": [
        "ui design","ux design","user interface","user experience","wireframing",
        "prototyping","figma","sketch","adobe xd","invision","zeplin",
        "design system","component library","typography","color theory",
        "accessibility","wcag","usability testing","user research",
        "information architecture","interaction design","visual design",
        "motion design","illustration","icon design","responsive design",
        "mobile first","design thinking","heuristic evaluation","a/b testing",
        "user journey","persona","empathy mapping","card sorting","tree testing",
        "adobe photoshop","adobe illustrator","adobe after effects","canva",
    ],
    "Business & Finance": [
        "financial analysis","financial modelling","forecasting","budgeting",
        "accounting","bookkeeping","taxation","auditing","risk analysis",
        "valuation","investment analysis","portfolio management","equity research",
        "trading","derivatives","options","fixed income","credit analysis",
        "business analysis","requirements gathering","gap analysis","process mapping",
        "business intelligence","market research","competitive analysis",
        "strategic planning","business development","sales","crm",
        "salesforce","hubspot","zendesk","erp","sap","oracle erp",
        "supply chain","logistics","procurement","inventory management",
        "operations management","six sigma","lean manufacturing","kaizen",
    ],
    "Soft Skills": [
        "communication","teamwork","leadership","problem solving","critical thinking",
        "time management","adaptability","creativity","collaboration","patience",
        "analytical thinking","attention to detail","logical reasoning","mentoring",
        "interpersonal skills","active listening","flexibility","empathy",
        "public speaking","presentation skills","negotiation","conflict resolution",
        "decision making","strategic thinking","innovation","entrepreneurship",
        "project coordination","cross-functional collaboration","stakeholder communication",
        "written communication","technical writing","documentation","reporting",
        "self-motivation","initiative","ownership","accountability","integrity",
        "growth mindset","continuous learning","resilience","stress management",
        "multitasking","prioritisation","organisation","planning","coaching",
        "training","onboarding","knowledge transfer","peer review","code review",
    ],
}

SKILL_TO_CAT     = {kw: cat for cat, kws in SKILL_ONTOLOGY.items() for kw in kws}
SOFT_SKILLS      = set(SKILL_ONTOLOGY["Soft Skills"])
ALL_KNOWN_SKILLS = {kw for kws in SKILL_ONTOLOGY.values() for kw in kws}

# ══════════════════════════════════════════════════════════════════════════════
# CANONICAL NORMALIZATION MAP
# ══════════════════════════════════════════════════════════════════════════════
CANONICAL = {
    "lesson plan":"lesson planning","lesson plans":"lesson planning",
    "curriculum planning":"curriculum development","curriculum plan":"curriculum development",
    "smart boards":"smart board",
    "online teaching platform":"online teaching","online teaching platforms":"online teaching",
    "classroom lecture":"classroom management","classroom lectures":"classroom management",
    "parent-teacher conference":"parent communication",
    "parent-teacher conferences":"parent communication",
    "parent teacher conference":"parent communication",
    "parent teacher conferences":"parent communication",
    "staff meeting":"staff meetings",
    "digital tool":"digital tools",
    "educational tool":"educational software",
    "problem-solving":"problem solving","problem-solving skills":"problem solving",
    "analytical":"analytical thinking","analytical skills":"analytical thinking",
    "time-management":"time management","timemanagement":"time management",
    "critical-thinking":"critical thinking",
    "rest apis":"rest api","restful api":"rest api","restful apis":"rest api",
    "restful":"rest api","node js":"node.js","nodejs":"node.js",
    "react js":"react","reactjs":"react",
    "scikit learn":"scikit-learn","sklearn":"scikit-learn",
    "pytorch lightning":"pytorch","torch":"pytorch","libtorch":"pytorch",
    "py torch":"pytorch","torchvision":"pytorch","torchaudio":"pytorch",
    "tensorflow 2":"tensorflow","tf2":"tensorflow",
    "tensorflow keras":"keras","tf.keras":"keras",
    "convolutional neural network":"cnn","conv neural network":"cnn","convolutional network":"cnn",
    "recurrent neural network":"rnn","recurrent network":"rnn",
    "long short term memory":"lstm","long short-term memory":"lstm",
    "gated recurrent unit":"gru",
    "large language model":"llm","large language models":"llm",
    "foundation model":"llm","foundation models":"llm",
    "huggingface":"hugging face","hf transformers":"hugging face",
    "hugging face transformers":"transformers",
    "xgb":"xgboost","lgbm":"lightgbm","light gbm":"lightgbm","cat boost":"catboost",
    "generative adversarial network":"gan","generative adversarial networks":"gan",
    "diffusion models":"diffusion model","gen ai":"generative ai","genai":"generative ai",
    "wandb":"weights and biases","w&b":"weights and biases","ml flow":"mlflow",
    "cudnn":"cuDNN","cuda toolkit":"cuda",
    "pyspark":"apache spark","postgres":"postgresql","mongo":"mongodb",
    "k8s":"kubernetes","gcp":"google cloud","ci cd":"ci/cd","cicd":"ci/cd",
    "nlp":"natural language processing","open cv":"opencv",
    "ms office":"ms office","microsoft office":"ms office",
    "microsoft azure":"azure","amazon web services":"aws",
    "b.e":None,"b.ed":None,"b.ed.":None,
    "b.e (engineering)":None,"b.ed (education)":None,
    "raspberry pi":"raspberry pi","raspberrypi":"raspberry pi",
    "packet tracer":"packet tracer","cisco packet tracer":"packet tracer",
    "networking fundamentals":"networking","networking basics":"networking",
    "basic security concepts":"cybersecurity","basic cybersecurity":"cybersecurity",
    "cybersecurity security":"cybersecurity",
    "embedded systems":"embedded systems",
    "continuouslearner":None,"softskills":None,
    "rest":None,"scalable workflow":None,"scalable workflows":None,
    "academic records":None,"academic record":None,"parent teacher":None,
    "problem skills patience":None,"patience and adaptability":"adaptability",
    "adaptability critical":None,"online teaching familiarity":None,
    "mathematics concepts":None,"standard":None,"familiarity":None,
    "concepts":None,"records":None,"academic":None,"critical":None,
    "different force fields":None,"differential equations visualization":None,
    "experimental datasets":None,"experimental results":None,"experimental data":None,
    "historical climate datasets":None,"historical climate data":None,
    "origin lab databases":None,"science majors":None,"sample science":None,
    "computational physics":None,"electromagnetism":None,"statistical mechanics":None,
    "statistical distributions":None,"temperature trends":None,"different force":None,
    "force fields":None,"climate datasets":None,"undergraduate":None,
    "analytical problem":"analytical thinking","predictive model":"machine learning",
    "prediction accuracy":None,"scikit":"scikit-learn",
    "jupyter notebook":"jupyter notebook","regression":"regression",
    "originlab":None,"origin lab":None,"cc++":None,"jan apr":None,
    "learn":None,"machine":None,"parameter":None,"particle":None,
    "pattern":None,"trajectory":None,"interest":None,"science":None,
    "physics":None,"drama":None,"madra":None,"scientific research":None,
    "data science":"machine learning",
}

SECTION_HEADERS = {
    "technical skills","soft skills","skills","projects","mini projects",
    "education","certifications","profile","experience","achievements",
    "activities","responsibilities","qualifications","competencies",
    "overview","summary","objective","references","declaration",
    "extra curricular activities","technical","personal",
}

HARD_NOISE = {
    "skill","ability","requirement","responsibility","qualification",
    "competency","candidate","applicant","objective","overview",
    "description","title","position","role","employment","type",
    "section","record","item","context","way","part","example","set","use",
    "strong","excellent","good","great","effective","ideal","preferred",
    "required","proficient","familiar","experienced","demonstrated",
    "proven","solid","seeking","looking","hiring","working","individual",
    "complex","dedicated","key","main","primary","various","specific",
    "multiple","additional","relevant","related","significant","higher",
    "conduct","manage","develop","create","ensure","provide","maintain",
    "support","implement","build","handle","participate","prepare",
    "coordinate","monitor","assist","deliver","facilitate","evaluate",
    "year","month","date","period","company","organization","department",
    "business","process","environment","opportunity","area","thing",
    "point","aspect","factor","result","output","number","name","value",
    "detail","team","product","group","committee","board","job",
    "resume","curriculum","vitae","reference","declaration","hobby",
    "nationality","contact","achievement","award","completion",
    "participation","volunteer","publication","paper","degree",
    "bachelor","master","cgpa","gpa","father","mother","gender",
    "b.e","b.ed","b.tech","m.tech","b.sc","m.sc","mba","bca","mca",
    "be","btech","mtech","bsc","msc","phd","doctorate",
    "religion","caste","class","batch","semester","certification",
    "bearer","extra","curricular","learner","outreach","execution",
    "scheme","service","national","institution","introduction",
    "foundation","fundamental","concept","basic","advanced",
    "door","vault","hand","hands","face","billing","advertisement",
    "english","practical","tutorial","continuous","efficient",
    "sri","xii","xiii","viii","vii","iii","college","university",
    "institute","school","kk","com","www","inc","ltd","pvt",
    "prof","dr","event","member","chapter","activity","program",
    "course","module","unit","session","report","associate",
    "intern","fresher","profile","secondary","professional",
    "engineer","technical","technician","specialist",
    "implementation","industry","intelligent","engagement",
    "reasoning","thinking","interactive","examination","test","tests",
    "grade","assignment","record","participate","office",
    "mini","soft","packet","tracer","introduce",
    "standard","familiarity","concepts","records","academic","critical",
    "familiar","skills","parent","teacher","patience","fragment",
    "learn","machine","parameter","particle","pattern","script",
    "trajectory","interest","science","physics","regression","drama",
    "madra","undergraduate","dataset","datasets","visualization",
    "electromagnetism","prediction","accuracy","origin","originlab",
    "force","field","fields","climate","temperature","trend","trends",
    "sample","majors","major","computational","experimental","historical",
    "differential","equations","equation","distribution","distributions",
}

SOFT_NOISE = {
    "data","smart","secure","open","new","system","network",
    "technology","information","computer","programming","language",
    "electronic","hardware","intelligence","industrial","internet",
    "recognition","simulation","checkout","training","management",
    "education","knowledge","experience","background","stack","queue",
    "structure","interface","control","access","cloud","platform",
    "level","entry","design","research","support","standard","resource",
    "tool","software","analysis","testing","monitoring","automation",
    "integration","solution","application","framework","library",
    "sensor","protocol","architecture","security","raspberry","mobile",
    "science","physics","chemistry","biology","academic","theoretical",
    "experimental","computational","statistical","numerical","simulation",
}

_ALWAYS_BLOCK = re.compile(
    r'^(https?|www\.|\.com|@|\d+|[a-z]{1,2}$|'
    r'jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|'
    r'january|february|march|april|june|july|august|september|october|november|december|'
    r'chennai|mumbai|delhi|bangalore|bengaluru|hyderabad|pune|kolkata|'
    r'madras|madurai|coimbatore|trichy|salem|tirunelveli|vellore|'
    r'kochi|thiruvananthapuram|mysore|mangalore|ahmedabad|surat|jaipur|'
    r'lucknow|kanpur|nagpur|indore|bhopal|patna|vadodara|'
    r'hackerrank|linkedin|github|gmail|kkconstruction|'
    r'mepco|sankara|schlenk|kovilpatti|kanchi|shofia|infosy|'
    r'nptel|ieee|nss|nssc|srm|vit|anna|amrita|sastra|'
    r'kcet|naac|ugc|aicte|nba|infosys|wipro|tcs|accenture)$',
    re.IGNORECASE
)

_NO_STRIP = {
    "calculus","access","status","campus","bonus","corpus","nexus",
    "radius","census","focus","locus","abacus","syllabus","nucleus",
    "axis","basis","thesis","analysis","emphasis","crisis","diagnosis",
    "hypothesis","synthesis","css","aws","ios","sass","c++","c#",
    "javascript","typescript","express","mongoose","kubernetes",
    "pandas","numpy","mathematics","physics","ethics","electronics",
    "mechanics","statistics","algorithms",
}


# ══════════════════════════════════════════════════════════════════════════════
# OFFLINE TF-IDF SIMILARITY  (replaces SentenceTransformer / BERT)
# ══════════════════════════════════════════════════════════════════════════════
def _tfidf_similarity(texts_a: list, texts_b: list) -> np.ndarray:
    """Cosine similarity matrix between two lists of strings using TF-IDF."""
    if not texts_a or not texts_b:
        return np.zeros((len(texts_a), len(texts_b)))
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    vectorizer.fit(texts_a + texts_b)
    vecs_a = vectorizer.transform(texts_a)
    vecs_b = vectorizer.transform(texts_b)
    return cosine_similarity(vecs_a, vecs_b)


# ══════════════════════════════════════════════════════════════════════════════
# NAME EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
_NAME_WORD_RE = re.compile(r'^[A-Za-z][a-zA-Z\-]{1,}$')

def _is_name_word(w: str) -> bool:
    if not _NAME_WORD_RE.match(w):
        return False
    wl = w.lower()
    if not (w[0].isupper() or w.isupper()):
        return False
    return (wl not in SECTION_HEADERS
            and wl not in ALL_KNOWN_SKILLS
            and wl not in HARD_NOISE)

def extract_candidate_name(raw_text: str) -> str:
    for line in raw_text.splitlines()[:20]:
        line = line.strip().lstrip('•-–*▪◦·|').strip()
        if not line or len(line) < 3:
            continue
        segments = re.split(r'[\|,;\t]|  +', line)
        for seg in segments:
            seg = seg.strip()
            words = seg.split()
            if not (2 <= len(words) <= 4):
                continue
            if not all(_is_name_word(w) for w in words):
                continue
            if re.search(r'[\d@\.:/]', seg):
                continue
            if words[0][0].isupper() and words[1][0].isupper():
                return seg.title()
    return ""

def _name_tokens(name: str) -> set:
    if not name:
        return set()
    return {w.lower() for w in name.split() if len(w) > 2}

def strip_name_from_text(raw_text: str, name: str) -> str:
    if not name:
        return raw_text
    name_lower = name.lower()
    kept = []
    for line in raw_text.splitlines():
        if name_lower in line.lower():
            kept.append("")
        else:
            kept.append(line)
    return "\n".join(kept)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TEXT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
_HEADER_LINE = re.compile(
    r'^(location|position|employment\s*type|job\s*type|job\s*title|'
    r'department|experience\s*required|salary|compensation|'
    r'reporting|shift|work\s*mode|notice\s*period|joining|ctc|'
    r'package|vacancies?|opening)\s*[:\|]',
    re.IGNORECASE
)
_BOILERPLATE = re.compile(
    r'(job description|job overview|position title|employment type|'
    r'key responsibilities|key requirements|key skills|key competencies|'
    r'preferred skills|required qualifications|'
    r'we are (looking|seeking|hiring)|the ideal candidate|'
    r'qualifications?\s*(required|preferred)|responsibilities include|'
    r'about (the|this) (role|position)|equal opportunity|how to apply|'
    r'career objective|professional summary|personal (information|details)|'
    r'date of birth|i hereby declare|languages? known|marital status|'
    r'father.{0,4}name|permanent address)',
    re.IGNORECASE
)

def preprocess_for_nlp(text: str) -> str:
    text = re.sub(r'https?\s*:\/\/\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\+?\d[\d\-\s]{8,}\d', ' ', text)
    text = re.sub(r'\b\d+\s?%', ' ', text)
    text = re.sub(r'cgpa\s*\d+(\.\d+)?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{4}\s*[-–]\s*\d{4}\b', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or _BOILERPLATE.search(s): continue
        if _HEADER_LINE.match(s): continue
        if re.match(r'^[\w\s\/\-]{1,50}:\s*$', s): continue
        words = s.split()
        if len(words) <= 4 and not any(c in s for c in ['.', ',', ';']):
            if words and words[0].lower() in SECTION_HEADERS:
                continue
        lines.append(s)
    text = ' '.join(lines)
    text = re.sub(r',\s*', '. ', text)
    text = re.sub(r';\s*', '. ', text)
    text = re.sub(r'(?<=[a-z])\s+and\s+(?=[a-z])', '. ', text)
    text = re.sub(r'(?<=[a-z])\s+or\s+(?=[a-z])', '. ', text)
    text = re.sub(r'\b\d+(\.\d+)?\b', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'[^a-zA-Z0-9\s\+\#\.\-\/]', ' ', text.lower())
    return re.sub(r'\s+', ' ', text).strip()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4a — STRUCTURED SECTION EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════
def extract_structured_skills(text: str, blocked_tokens: set = None) -> set:
    blocked_tokens = blocked_tokens or set()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    skills = set()

    for line in text.splitlines():
        line = line.strip().lstrip('•-–*▪◦·').strip()
        line = re.sub(r'\([^)]*\d{4}[^)]*\)', '', line).strip()
        line = re.sub(r'\b\d{4}\b', '', line).strip()

        if not line or len(line) < 2:
            continue
        line_lower = line.lower()
        if line_lower in SECTION_HEADERS:
            continue
        if _BOILERPLATE.search(line):
            continue

        if ':' in line:
            cat, _, items_str = line.partition(':')
            cat_lower = cat.strip().lower()
            if cat_lower in SECTION_HEADERS:
                continue
            for item in items_str.split(','):
                item = item.strip()
                item = re.sub(r'\([^)]*\)', '', item).strip()
                if not item or len(item) < 2: continue
                if re.search(r'\d', item): continue
                if '@' in item or '.com' in item: continue
                if item.lower() in blocked_tokens: continue
                norm = _normalize_one(item.lower())
                if norm and norm not in blocked_tokens:
                    skills.add(norm)
        else:
            if 2 <= len(line) <= 50 and len(line.split()) <= 4:
                if re.search(r'\d', line): continue
                if '@' in line or '.com' in line: continue
                if line_lower in blocked_tokens: continue
                norm = _normalize_one(line_lower)
                if norm and norm not in blocked_tokens:
                    skills.add(norm)

    return skills


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4b — NLP EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════
MAX_PHRASE_WORDS = 3

def _tok_ok(tok, blocked_tokens: set = None) -> bool:
    blocked_tokens = blocked_tokens or set()
    txt = tok.text.lower()
    return (
        tok.pos_ in ("NOUN", "PROPN", "ADJ")
        and not tok.is_stop
        and txt not in HARD_NOISE
        and txt not in blocked_tokens
        and not _ALWAYS_BLOCK.match(txt)
        and len(txt) >= 3
        and not re.match(r'^\d+$', txt)
        and '.com' not in txt
    )

def _phrase_valid(tokens) -> bool:
    return (
        all(
            t.text.lower() not in HARD_NOISE
            and t.text.lower() not in SOFT_NOISE
            and not _ALWAYS_BLOCK.match(t.text.lower())
            for t in tokens
        )
        and not any(t.pos_ == "VERB" for t in tokens)
    )

def extract_nlp_skills(text: str, blocked_tokens: set = None) -> set:
    blocked_tokens = blocked_tokens or set()
    doc = nlp(text)
    ner_noise = set()
    ner_text_blocks = set()
    for ent in doc.ents:
        if ent.label_ in ("PERSON","GPE","LOC","ORG","DATE","TIME",
                           "CARDINAL","ORDINAL","MONEY","PERCENT",
                           "QUANTITY","NORP","FAC","EVENT","WORK_OF_ART"):
            for t in ent:
                ner_noise.add(t.i)
            ner_text_blocks.add(ent.text.lower())

    seen, results, chunk_idx = set(), [], set()

    for chunk in doc.noun_chunks:
        tokens = [t for t in chunk if t.i not in ner_noise and _tok_ok(t, blocked_tokens)]
        if not tokens: continue
        if len(tokens) > MAX_PHRASE_WORDS: continue
        if not _phrase_valid(tokens):
            for t in tokens: chunk_idx.add(t.i)
            continue
        phrase = " ".join(t.text.lower() for t in tokens)
        if phrase in blocked_tokens: continue
        if phrase not in seen:
            seen.add(phrase); results.append(phrase)
            for t in tokens: chunk_idx.add(t.i)

    for tok in doc:
        if tok.i in ner_noise or tok.i in chunk_idx: continue
        if tok.pos_ in ("NOUN","PROPN") and _tok_ok(tok, blocked_tokens):
            w = tok.text.lower()
            if w not in blocked_tokens and w not in seen:
                seen.add(w); results.append(w)

    return {_normalize_one(r) for r in results if _normalize_one(r)} - ner_text_blocks


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════
def _normalize_one(phrase: str):
    p = phrase.lower().strip()
    if not p or len(p) < 2: return None
    if p in HARD_NOISE: return None
    if _ALWAYS_BLOCK.match(p): return None
    if p in CANONICAL:
        return CANONICAL[p]
    if p.endswith('ies') and len(p) > 5 and ' ' not in p:
        p = p[:-3] + 'y'
    if (' ' not in p and p.endswith('s') and len(p) > 5
            and p not in _NO_STRIP
            and not re.match(r'^[a-z]{2,4}$', p)):
        singular = p[:-1]
        if not re.search(r'[aeiou][bcdfghjklmnpqrtvwxyz]{3,}$', singular):
            p = singular
    if p in HARD_NOISE: return None
    return p

def split_co_listed(phrase: str) -> list:
    words = phrase.split()
    if len(words) >= 2 and all(w in ALL_KNOWN_SKILLS for w in words):
        return words
    return [phrase]

def finalize_skills(raw: set, source_text: str = "", blocked_tokens: set = None) -> set:
    blocked_tokens = blocked_tokens or set()
    normed = set()

    for s in raw:
        if not s: continue
        s = s.lower().strip()
        if re.search(r'\d', s): continue
        if '@' in s or '.com' in s or 'http' in s: continue
        if len(s) < 3: continue
        if s in HARD_NOISE: continue
        if _ALWAYS_BLOCK.match(s): continue
        if s in blocked_tokens: continue
        normed.update(split_co_listed(s))

    normed = {s for s in normed if len(s.split()) <= 3 and s not in blocked_tokens}
    confirmed = set()

    for s in normed:
        if s in ALL_KNOWN_SKILLS:
            confirmed.add(s)

    if source_text:
        text_lower = source_text.lower()
        for skill in ALL_KNOWN_SKILLS:
            if len(skill) < 4 and ' ' not in skill:
                if re.search(r'(?<![a-zA-Z])' + re.escape(skill) + r'(?![a-zA-Z])', text_lower):
                    confirmed.add(skill)
            else:
                if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                    confirmed.add(skill)

    confirmed = {s for s in confirmed if s not in blocked_tokens}

    to_remove = set()
    for longer in sorted(confirmed):
        words = longer.split()
        if len(words) < 2: continue
        for n in range(1, len(words)):
            base = " ".join(words[:n])
            if base in confirmed:
                to_remove.add(longer)
                break
    confirmed -= to_remove

    return confirmed


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def extract_skills_from_resume(raw_text: str, blocked_tokens: set = None, name: str = "") -> set:
    blocked_tokens = blocked_tokens or set()
    clean_text  = strip_name_from_text(raw_text, name)
    structured  = extract_structured_skills(clean_text, blocked_tokens)
    prose_clean = preprocess_for_nlp(clean_text)
    nlp_skills  = extract_nlp_skills(prose_clean, blocked_tokens)
    return finalize_skills(structured | nlp_skills,
                           source_text=clean_text.lower(),
                           blocked_tokens=blocked_tokens)

def extract_skills_from_jd(raw_text: str) -> set:
    structured  = extract_structured_skills(raw_text)
    prose_clean = preprocess_for_nlp(raw_text)
    nlp_skills  = extract_nlp_skills(prose_clean)
    return finalize_skills(structured | nlp_skills, source_text=raw_text.lower())


# ══════════════════════════════════════════════════════════════════════════════
# STEPS 6–9
# ══════════════════════════════════════════════════════════════════════════════
def apply_ontology(skills):
    skill_cats = {}; cat_hits = defaultdict(list)
    for sk in skills:
        sl = sk.lower(); cat = SKILL_TO_CAT.get(sl)
        if not cat:
            for kw, c in SKILL_TO_CAT.items():
                if kw in sl or sl in kw: cat = c; break
        cat = cat or "Other Technical"
        skill_cats[sk] = cat; cat_hits[cat].append(sk)
    return skill_cats, dict(cat_hits)

def categorize_skills(skills):
    tech, soft = set(), set()
    for sk in skills:
        sl = sk.lower()
        (soft if sl in SOFT_SKILLS or any(s in sl for s in SOFT_SKILLS)
         else tech).add(sk)
    return tech, soft

def semantic_match(jd_skills, res_skills, threshold=0.72):
    if not jd_skills or not res_skills:
        return [], list(jd_skills), list(res_skills)
    jd_l  = list(jd_skills)
    res_l = list(res_skills)
    sim   = _tfidf_similarity(jd_l, res_l)
    matched, used = [], set()
    for j, jsk in enumerate(jd_l):
        bi = int(sim[j].argmax())
        bv = float(sim[j][bi])
        if bv >= threshold:
            matched.append({"jd_skill": jsk, "resume_skill": res_l[bi],
                            "similarity": round(bv, 3)})
            used.add(bi)
    matched_jd = {m["jd_skill"] for m in matched}
    return (matched,
            [s for s in jd_l  if s not in matched_jd],
            [res_l[i] for i in range(len(res_l)) if i not in used])

def compute_score(matched, jd_skills, jd_clean, res_clean):
    total   = len(jd_skills) or 1
    sk_pct  = len(matched) / total * 100
    sim     = _tfidf_similarity([jd_clean[:512]], [res_clean[:512]])
    cos_pct = float(sim[0][0]) * 100
    return round(0.55*sk_pct + 0.45*cos_pct, 2), round(sk_pct, 2), round(cos_pct, 2)


# ── Domain detection ──────────────────────────────────────────────────────────
DOMAIN_PROFILES = {
    "Machine Learning / AI": {
        "keywords": ["machine learning","deep learning","neural network","pytorch","tensorflow",
                     "keras","scikit-learn","nlp","computer vision","llm","bert","gpt","xgboost",
                     "reinforcement learning","data science","kaggle","hugging face","generative ai"],
        "color": "#7c3aed", "bg": "#f5f3ff", "border": "#c4b5fd",
    },
    "Web Development": {
        "keywords": ["react","angular","vue","node.js","django","flask","fastapi","html","css",
                     "javascript","typescript","rest api","next.js","graphql","tailwind","bootstrap"],
        "color": "#0277b5", "bg": "#e0f5ff", "border": "#b3e5fc",
    },
    "Data Engineering": {
        "keywords": ["etl","elt","apache spark","kafka","airflow","dbt","hadoop","bigquery",
                     "snowflake","redshift","databricks","data pipeline","data warehouse","pyspark"],
        "color": "#0f766e", "bg": "#f0fdfa", "border": "#99f6e4",
    },
    "Cloud & DevOps": {
        "keywords": ["aws","azure","gcp","docker","kubernetes","terraform","ci/cd","jenkins",
                     "ansible","devops","linux","helm","nginx","cloud"],
        "color": "#b45309", "bg": "#fffbeb", "border": "#fde68a",
    },
    "Embedded Systems": {
        "keywords": ["arduino","raspberry pi","stm32","esp32","rtos","firmware","fpga","verilog",
                     "microcontroller","embedded linux","iot","sensor","uart","spi","i2c"],
        "color": "#166534", "bg": "#f0fdf4", "border": "#bbf7d0",
    },
    "Networking & Security": {
        "keywords": ["networking","cybersecurity","tcp/ip","firewall","vpn","wireshark","cisco",
                     "penetration testing","packet tracer","information security"],
        "color": "#be123c", "bg": "#fff1f2", "border": "#fecdd3",
    },
    "Teaching & Education": {
        "keywords": ["lesson planning","classroom management","curriculum development","online teaching",
                     "student engagement","blended learning","smart board","educational software"],
        "color": "#9a3412", "bg": "#fff7ed", "border": "#fed7aa",
    },
    "Mobile Development": {
        "keywords": ["android","ios","flutter","react native","kotlin","swift","dart","mobile"],
        "color": "#1d4ed8", "bg": "#eff6ff", "border": "#bfdbfe",
    },
}

def detect_domain(skills: set, raw_text: str) -> list:
    text_lower = raw_text.lower()
    scores = {}
    for domain, profile in DOMAIN_PROFILES.items():
        kws = profile["keywords"]
        skill_hits = sum(1 for k in kws if k in skills)
        text_hits  = sum(1 for k in kws if k in text_lower)
        total_hits = skill_hits * 2 + text_hits
        if total_hits > 0:
            scores[domain] = round(min(total_hits / len(kws) * 100, 99))
    return sorted([{"domain": d, "score": s, **DOMAIN_PROFILES[d]}
                   for d, s in scores.items()], key=lambda x: -x["score"])[:3]


# ══════════════════════════════════════════════════════════════════════════════
# PROJECT COUNT — section-aware
# ══════════════════════════════════════════════════════════════════════════════
def count_projects(raw_text: str) -> int:
    lines = raw_text.splitlines()
    in_project_section = False
    project_count = 0
    last_was_blank = True  # treat start of section as blank

    PROJECT_SECTION_RE = re.compile(
        r'^(projects?|mini[\s\-]?projects?|academic\s*projects?|'
        r'personal\s*projects?|major\s*projects?|minor\s*projects?|'
        r'key\s*projects?|project\s*work|project\s*details?)\s*$',
        re.IGNORECASE
    )
    OTHER_SECTION_RE = re.compile(
        r'^(education|certifications?|experience|skills?|achievements?|'
        r'activities|responsibilities|references?|declaration|'
        r'extra\s*curricular|summary|objective|profile|competencies?)\s*$',
        re.IGNORECASE
    )

    # Lines that are clearly descriptions/metadata, NOT project titles
    DESCRIPTION_SIGNALS = re.compile(
        r'^(technologies?|tech\s*stack|tools?|languages?|frameworks?|'
        r'duration|period|role|platform|database|frontend|backend|'
        r'description|objective|features?|modules?|responsibilities)\s*[:\-]',
        re.IGNORECASE
    )

    for line in lines:
        stripped = line.strip()

        # Entering project section
        if PROJECT_SECTION_RE.match(stripped):
            in_project_section = True
            last_was_blank = True
            continue

        # Leaving project section
        if in_project_section and OTHER_SECTION_RE.match(stripped):
            in_project_section = False
            continue

        if not in_project_section:
            continue

        # Track blank lines — a blank line signals a new project entry is coming
        if not stripped:
            last_was_blank = True
            continue

        # Skip description/metadata lines
        if DESCRIPTION_SIGNALS.match(stripped):
            last_was_blank = False
            continue

        # Skip lines that are clearly descriptions (long sentences)
        if len(stripped.split()) > 12:
            last_was_blank = False
            continue

        # Skip lines starting with action verbs (description bullets)
        if re.match(r'^(developed|built|designed|implemented|created|used|'
                    r'integrated|deployed|worked|performed|analysed|applied|'
                    r'utilized|leveraged|responsible|involved)\b',
                    stripped, re.IGNORECASE):
            last_was_blank = False
            continue

        # A numbered entry like "1." or "1)" always = one project
        if re.match(r'^\d+[\.\)]\s+\S', stripped):
            project_count += 1
            last_was_blank = False
            continue

        # A bullet point — count only if it looks like a title (short, capitalised)
        if re.match(r'^[\•\-\*\➢\➤\▪\◦\·\u2022\u2023\u25aa]\s*\S', stripped):
            content = re.sub(r'^[\•\-\*\➢\➤\▪\◦\·\u2022\u2023\u25aa]\s*', '', stripped)
            words = content.split()
            # Title-like: short (≤8 words), starts with capital, no trailing period
            if (len(words) <= 8
                    and words[0][0].isupper()
                    and not content.endswith('.')
                    and not re.match(r'^(developed|built|designed|implemented|'
                                     r'created|used|integrated|deployed)', content, re.IGNORECASE)):
                project_count += 1
            last_was_blank = False
            continue

        # Plain line after a blank — likely a project title
        if last_was_blank:
            words = stripped.split()
            if (1 <= len(words) <= 8
                    and stripped[0].isupper()
                    and not stripped.endswith('.')
                    and not re.search(r'\d{4}', stripped)):
                project_count += 1

        last_was_blank = False

    # Fallback
    if project_count == 0:
        matches = re.findall(
            r'\b(?:developed|built|designed|implemented|created|deployed'
            r'|constructed|engineered|architected)\b',
            raw_text.lower()
        )
        project_count = len(set(matches)) if matches else 0

    return min(project_count, 20)


# ══════════════════════════════════════════════════════════════════════════════
# RESUME INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
"""
DROP-IN REPLACEMENT for extract_resume_insights() in app.py
Replace the entire existing function with this one.

Key fix: works on FLAT single-line PDF text (what fitz produces).
Instead of line-by-line parsing, it splits on bullet chars (•▪◦·)
and tracks the current section by detecting section-name suffixes
appended to the end of each chunk.
"""

def extract_resume_insights(raw_text: str, res_skills: set, res_tech: list, res_soft: list) -> dict:
    text  = raw_text.lower()

    # ── Years of experience ──────────────────────────────────────────────────
    years_exp = None
    for pat in [
        r'(\d+)\+?\s*years?\s*of\s*(?:work\s*)?experience',
        r'(\d+)\+?\s*years?\s*(?:experience|exp)',
        r'experience\s*(?:of\s*)?(\d+)\+?\s*years?',
    ]:
        m = re.search(pat, text)
        if m:
            years_exp = int(m.group(1)); break

    # ── Education level ──────────────────────────────────────────────────────
    edu_level = "Not specified"
    if any(w in text for w in ["ph.d","phd","doctorate","doctoral"]):
        edu_level = "PhD"
    elif any(w in text for w in ["m.tech","mtech","m.e","m.sc","msc","m.s","master","mba","m.b.a"]):
        edu_level = "Postgraduate"
    elif any(w in text for w in ["b.tech","btech","b.e","be","b.sc","bsc","b.s","bachelor","undergraduate","b.com","bca"]):
        edu_level = "Undergraduate"
    elif any(w in text for w in ["diploma","polytechnic"]):
        edu_level = "Diploma"
    elif any(w in text for w in ["12th","xii","higher secondary","hsc","plus two"]):
        edu_level = "12th / HSC"

    # ── Bullet-chunk walker ──────────────────────────────────────────────────
    # PDF text is often one long line. We split on bullet chars and walk the
    # chunks, detecting section-name suffixes to track the current section.
    # e.g. "...MATLAB Mini projects Certifications" ends a tools bullet and
    # marks the start of the Certifications section.

    _tail_section_re = re.compile(
        r'\s+(Mini\s+[Pp]rojects?|[Pp]rojects?|Certifications?|Certificates?|'
        r'Technical\s+Skills?|Soft\s+Skills?|Skills?|'
        r'Education|Experience|Work\s+Experience|Profile|Summary|'
        r'Achievements?|Extra\s+Curricular(?:\s+Activities)?|Activities|'
        r'Paper\s+[Pp]resented|Declaration|References?|Internship|'
        r'Courses?|Training|Workshops?|Languages?|Hobbies)\s*$'
    )

    # Signals that a bullet belongs to a project, not a certification
    _proj_hw = re.compile(
        r'\b(?:arduino|raspberry\s*pi|rfid|opencv|biometric|'
        r'queue|stack|data\s+struct|checkout|door\s+access|'
        r'smart\s+vault|billing|management\s+system|embedded\s+hardware|'
        r'embedded\s+hardware|face.{0,5}recogni)\b',
        re.IGNORECASE
    )
    # Strip action-verb descriptions appended inline
    _action_split_re = re.compile(
        r'\s+(?=(?:Implemented|Designed|Integrated|Developed|Built|Created|'
        r'Used|Configured|Conducted|Performed|Achieved|Managed|'
        r'Led|Worked|Assisted|Supported|Presented)\b)'
    )
    # Clean trailing date from key
    _date_clean = re.compile(
        r'\s*\((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+20\d\d\)\s*$'
        r'|\s*[-–]\s*20\d\d\s*$',
        re.IGNORECASE
    )

    current_section = "header"
    cert_bullets = []
    proj_bullets = []

    # Support both • and newline-separated resumes
    # First try splitting on bullets; if only 1 chunk, fall back to lines
    chunks = re.split(r'\s*[•▪◦·]\s*', raw_text)
    if len(chunks) <= 2:
        chunks = raw_text.splitlines()

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        # Detect a section header at the tail of this chunk
        tail_m = _tail_section_re.search(chunk)
        if tail_m:
            bullet_content = chunk[:tail_m.start()].strip()
            new_section    = tail_m.group(1).strip().lower()
            new_section    = re.sub(r'\s+', ' ', new_section)
        else:
            bullet_content = chunk
            new_section    = None

        # File the bullet content under the current section
        if bullet_content and len(bullet_content) > 4:
            sec = current_section.lower()
            if re.match(r'certifications?|certificates?', sec):
                cert_bullets.append(bullet_content)
            elif re.match(r'(?:mini\s+)?projects?', sec):
                proj_bullets.append(bullet_content)

        if new_section:
            current_section = new_section

    # ── Classify cert bullets into certs vs projects ─────────────────────────
    def make_key(title):
        key = _date_clean.sub('', title).strip().lower()
        return re.sub(r'\s+', ' ', key).strip()

    certs = set()
    projs_from_cert = set()

    for b in cert_bullets:
        # Split inline description ("Title - 2025 Designed...")
        parts = _action_split_re.split(b, maxsplit=1)
        title = parts[0].strip()
        if not title or len(title) < 5:
            continue
        key = make_key(title)
        if not key or len(key) < 5:
            continue
        if _proj_hw.search(title):
            projs_from_cert.add(key[:70])
        else:
            certs.add(key[:80])

    all_projs = projs_from_cert.copy()
    for b in proj_bullets:
        parts = _action_split_re.split(b, maxsplit=1)
        title = parts[0].strip()
        if not title or len(title) < 5:
            continue
        key = make_key(title)
        if key and len(key) > 4:
            all_projs.add(key[:70])

    cert_count    = min(len(certs),     20)
    project_count = min(len(all_projs), 20)

    # ── Skill breadth ────────────────────────────────────────────────────────
    cats_covered = len({SKILL_TO_CAT.get(s) for s in res_skills if SKILL_TO_CAT.get(s)})
    total_cats   = len(SKILL_ONTOLOGY)
    breadth_pct  = round(cats_covered / total_cats * 100)

    return {
        "years_exp":     years_exp,
        "edu_level":     edu_level,
        "project_count": project_count,
        "cert_count":    cert_count,
        "skill_breadth": breadth_pct,
        "tech_count":    len(res_tech),
        "soft_count":    len(res_soft),
        "cats_covered":  cats_covered,
    }

# ══════════════════════════════════════════════════════════════════════════════
# TF-IDF SIMILARITY (FOR FULL MATRIX)
# ══════════════════════════════════════════════════════════════════════════════
def _tfidf_similarity(list1, list2):
    if not list1 or not list2:
        return [[0]*len(list2) for _ in range(len(list1))]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(list1 + list2)

    jd_vecs = vectors[:len(list1)]
    res_vecs = vectors[len(list1):]

    return cosine_similarity(jd_vecs, res_vecs)

# ══════════════════════════════════════════════════════════════════════════════
# FILE HANDLING
# ══════════════════════════════════════════════════════════════════════════════
def allowed_file(f):
    return '.' in f and f.rsplit('.',1)[1].lower() in ALLOWED

def extract_text(file):
    fn = file.filename.lower()
    if not allowed_file(fn): return None
    try:
        if fn.endswith(".pdf"):
            b = file.read()
            return "".join(p.get_text("text") for p in fitz.open(stream=b, filetype="pdf"))
        elif fn.endswith(".docx"):
            return " ".join(p.text for p in Document(file).paragraphs)
        elif fn.endswith(".txt"):
            return file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print("Read error:", e)
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(jd_raw, resume_raw):
    candidate_name = extract_candidate_name(resume_raw)
    blocked_tokens = _name_tokens(candidate_name)

    jd_skills  = extract_skills_from_jd(jd_raw)
    res_skills = extract_skills_from_resume(resume_raw, blocked_tokens, name=candidate_name)

    _, jd_cats  = apply_ontology(jd_skills)
    _, res_cats = apply_ontology(res_skills)

    jd_tech,  jd_soft  = categorize_skills(jd_skills)
    res_tech, res_soft = categorize_skills(res_skills)

    m_t, x_t, e_t = semantic_match(jd_tech,  res_tech)
    m_s, x_s, e_s = semantic_match(jd_soft,  res_soft)
    matched = m_t + m_s
    missing = x_t + x_s
    extra   = e_t + e_s

    jd_clean  = preprocess_for_nlp(jd_raw)
    res_clean = preprocess_for_nlp(resume_raw)
    final, sk_pct, cos_pct = compute_score(matched, jd_skills, jd_clean, res_clean)

    # ✅ FULL MATRIX (FIXED INDENTATION)
    jd_list = list(jd_skills)
    res_list = list(res_skills)

    try:
        full_matrix = _tfidf_similarity(jd_list, res_list)
        full_matrix = full_matrix.tolist()
    except:
        full_matrix = [[0]*len(res_list) for _ in range(len(jd_list))]
    # ===== RADAR VALUES (CATEGORY WISE) =====
    jd_tech_score = 100
    res_tech_score = round((len(res_tech) / len(jd_tech)) * 100, 2) if jd_tech else 0

    jd_soft_score = 100
    res_soft_score = round((len(res_soft) / len(jd_soft)) * 100, 2) if jd_soft else 0

    jd_sem_score = 100
    res_sem_score = cos_pct

    jd_gap_score = 100
    res_gap_score = round((len(matched) / (len(matched)+len(missing))) * 100, 2) if (matched or missing) else 0

    jd_dom_score = 100
    res_dom_score = final

    
    return {
        "score": final,
        "skill_score": sk_pct,
        "cosine_score": cos_pct,

        "full_jd_skills": jd_list,
        "full_res_skills": res_list,
        "full_matrix": full_matrix,

        "candidate_name": candidate_name,
        "resume_text": resume_raw.strip(),
        "jd_text": jd_raw.strip(),
        "resume_clean": res_clean.strip(),
        "jd_clean": jd_clean.strip(),

        "jd_technical": sorted(jd_tech),
        "jd_soft": sorted(jd_soft),
        "resume_technical": sorted(res_tech),
        "resume_soft": sorted(res_soft),

        "matched_skills": sorted(matched, key=lambda x: x["jd_skill"]),
        "missing_skills": sorted(missing),
        "extra_skills": sorted(extra),

        "jd_categories": jd_cats,
        "res_categories": res_cats,

        "insights": extract_resume_insights(
            resume_raw, res_skills, list(res_tech), list(res_soft) ),
        "radar": {

            "jd": [jd_tech_score, jd_soft_score, jd_sem_score, jd_gap_score, jd_dom_score],
            "resume": [res_tech_score, res_soft_score, res_sem_score, res_gap_score, res_dom_score]
        },
       
    }


def analyze_multiple(resume_texts, resume_names, jd_text):
    out = []
    for rt, name in zip(resume_texts, resume_names):
        r = run_pipeline(jd_text, rt)
        r["filename"] = name
        out.append(r)
    return sorted(out, key=lambda x: x["score"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        if (request.form.get('username') == "athithi"
                and request.form.get('password') == "athithi"):
            session['user'] = "athithi"
            return redirect(url_for('index'))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))
    resumes = request.files.getlist('resume')
    jd      = request.files.get('jd')
    if not resumes or not jd:
        flash("Upload JD and at least one resume.")
        return redirect(url_for('index'))
    if not allowed_file(jd.filename):
        flash("Invalid JD file type.")
        return redirect(url_for('index'))
    jd_text = extract_text(jd)
    if not jd_text:
        flash("Could not read JD file.")
        return redirect(url_for('index'))
    resume_texts, resume_names = [], []
    for res in resumes:
        if not allowed_file(res.filename):
            flash(f"Invalid: {res.filename}")
            return redirect(url_for('index'))
        text = extract_text(res)
        if not text:
            flash(f"Could not read: {res.filename}")
            return redirect(url_for('index'))
        resume_texts.append(text)
        resume_names.append(res.filename)
    results = analyze_multiple(resume_texts, resume_names, jd_text)
    return render_template("index.html", results=results)


if __name__ == "__main__":
    print(f"=== Resume Analyzer {APP_VERSION} starting ===")
    app.run(debug=True)

