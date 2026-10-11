# Retrieval-grading spot check (20 items)

Grades were produced blind by two Claude graders plus an adjudicator. For each context: relevance / sufficient / grader verdict. Gold = SHROOM-CAP factuality label (1 = has factual mistakes).

## en-tst-179  (gold factual mistake = 0)

**Question:** What is the full form of MDP?

**Answer under evaluation:** (The article can be found at <https://arxiv.org/abs/2006.14798>)The full form of MDP is Markov Decision Process.

- **pseudo_rag** → relevance `answers`, sufficient `True`, verdict `correct` — _Markov decision process (MDP) explicit; arXiv link unverifiable from context._
  > This formulation of an iterative retriever naturally fits in the definition of a Markov decision process (MDP).

- **bm25_k3** → relevance `partial`, sufficient `True`, verdict `correct` — _'our MDP' and 'framed as Markov decision processes' co-occur; expansion inferable._
  > 7pm tonight]] MTOP When is my next staff meeting scheduled for? (Yield :output (Event.start :obj (FindNumNextEvent :constraint (Event.subject_? :obj (?~= “staff meeting”)) :number 1L))) CalFlow Hey assistant, what is the price range of Stazione restaurant? (plan (Find :focus (Restaurant.priceRange_? always) :object (Restaurant.restaurantName_? (?= “Stazione restaurant”)))) TreeDST Figure 3: Samples of(𝑥,𝑦) pairs for semantic parsing under different datasets used in this paper. • The state of the MDP, i.e. the sequence of exem- plars, is modeled by a fixed-length vector s∈ R𝑑. The initial state s0 is a parameter. • At each step 1 exemplar is retrieved. We define a policy [...] be seen as an a

- **dense_k3** → relevance `partial`, sufficient `False`, verdict `undecidable` — _'(MDP)' and 'decision process' present; 'Markov' cut off._
  > (MDP). Here, our decision process comprises of(D∗, D,𝜏,𝑟), where • The state set D∗ contains exemplar sequences whose elements are in D; • The action set is just D: each action selects one exemplar from D. In theory, more than 1 exem- plar can be selected at each step, but we proceed with just 1 exemplar for simplicity; • The transition function𝜏 : D∗×D→ D∗ appends an exemplar to the existing sequence; • The reward function 𝑟 : D∗× D→ R funnels signal from the LLM back to the retriever. It will be discussed in §4. By situating our proposed [...] 7pm tonight]] MTOP When is my next staff meeting scheduled for? (Yield :output (Event.start :obj (FindNumNextEvent :constraint (Event.subject_? :obj

## en-tst-82  (gold factual mistake = 1)

**Question:** How many subtasks does the memorizing task consist of?

**Answer under evaluation:** 10 subtasks for the memorizing task.

- **pseudo_rag** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Five subtasks, not 10_
  > The memorizing task consists of five subtasks, each probing the memorization of an ontological rela- tionship: (1) TP: types of a given instance, (2) SCO: superclasses of a given class, (3) SPO: su- perproperties of a given property, (4) DM: domain constraint on a given property, and (5) RG: range constraint on a given property.

- **bm25_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Five subtasks stated; answer says 10_
  > using property constraint (P2302) in Wikidata. (3) Align the Wikidata properties with DBpedia proper- ties by equivalent property (P1628). (4) Query the domain and range constraints of the properties in DBpedia. (5) Cleanse the collected constraints us- ing the above-collected class set as vocabulary. We choose 50 properties with sensible domain, range and superproperties. 2.2 Construction of Memorizing Task The memorizing task consists of five subtasks, each probing the memorization of an ontological rela- tionship: (1) TP: types of a given instance, (2) SCO: superclasses of a given class, (3) SPO: su- perproperties of a given property, (4) DM: domain [...] domain, range and superproperties

- **dense_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Five subtasks stated; answer says 10_
  > using property constraint (P2302) in Wikidata. (3) Align the Wikidata properties with DBpedia proper- ties by equivalent property (P1628). (4) Query the domain and range constraints of the properties in DBpedia. (5) Cleanse the collected constraints us- ing the above-collected class set as vocabulary. We choose 50 properties with sensible domain, range and superproperties. 2.2 Construction of Memorizing Task The memorizing task consists of five subtasks, each probing the memorization of an ontological rela- tionship: (1) TP: types of a given instance, (2) SCO: superclasses of a given class, (3) SPO: su- perproperties of a given property, (4) DM: domain [...] domain, range and superproperties

## en-tst-223  (gold factual mistake = 0)

**Question:** What is the full form of MDP?

**Answer under evaluation:** The full form of MDP in the article is "Markov Decision Process".

- **pseudo_rag** → relevance `answers`, sufficient `True`, verdict `correct`
  > This approach can be formulated as Markov de- cision processes (MDPs).

- **bm25_k3** → relevance `answers`, sufficient `True`, verdict `correct` — _Window 3 says 'framed as Markov decision processes' alongside 'our MDP'._
  > 7pm tonight]] MTOP When is my next staff meeting scheduled for? (Yield :output (Event.start :obj (FindNumNextEvent :constraint (Event.subject_? :obj (?~= “staff meeting”)) :number 1L))) CalFlow Hey assistant, what is the price range of Stazione restaurant? (plan (Find :focus (Restaurant.priceRange_? always) :object (Restaurant.restaurantName_? (?= “Stazione restaurant”)))) TreeDST Figure 3: Samples of(𝑥,𝑦) pairs for semantic parsing under different datasets used in this paper. • The state of the MDP, i.e. the sequence of exem- plars, is modeled by a fixed-length vector s∈ R𝑑. The initial state s0 is a parameter. • At each step 1 exemplar is retrieved. We define a policy [...] be seen as an a

- **dense_k3** → relevance `partial`, sufficient `False`, verdict `undecidable` — _Window starts at '(MDP)'; expansion cut off, only 'decision process' visible._
  > (MDP). Here, our decision process comprises of(D∗, D,𝜏,𝑟), where • The state set D∗ contains exemplar sequences whose elements are in D; • The action set is just D: each action selects one exemplar from D. In theory, more than 1 exem- plar can be selected at each step, but we proceed with just 1 exemplar for simplicity; • The transition function𝜏 : D∗×D→ D∗ appends an exemplar to the existing sequence; • The reward function 𝑟 : D∗× D→ R funnels signal from the LLM back to the retriever. It will be discussed in §4. By situating our proposed [...] 7pm tonight]] MTOP When is my next staff meeting scheduled for? (Yield :output (Event.start :obj (FindNumNextEvent :constraint (Event.subject_? :obj

## en-tst-385  (gold factual mistake = 1)

**Question:** What are the three tasks that were built to test how well AI models “understand” humor across vision and language ?

**Answer under evaluation:** I apologize, but there is no article titled "Do Androids Laugh at Electric Sheep? Humor “Understanding” Benchmarks from The New Yorker Caption Contest" by Hessel, Jack et al.It's possible that you may have come across a fictional or hypothetical article, or that the article does not exist. If you can provide more context or information about the article, I may be able to help you better.However, if you're interested in learning about AI models and humor, there are several research papers and projects that have explored the topic. For example, the "WIT" (Wit Intelligence Test) is a dataset and 

- **pseudo_rag** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Three tasks listed; answer denies paper exists._
  > We develop a suite of three progressively harder tasks built around this contest to test how well AI models “understand” humor across vision and language: 1) matching jokes to cartoons, 2) identifying a winning caption, and 3) generating an explanation of why an image/caption combination is funny.

- **bm25_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _All windows list the tasks; refusal counts as incorrect._
  > they really “understand” humor? We challenge AI models with three tasks derived from the New Yorker Cartoon Caption Con- test: matching a joke to a cartoon, identify- ing a winning caption, and explaining why a winning caption is funny. These tasks en- capsulate progressively more sophisticated as- pects of “understanding” a cartoon; key ele- ments are the complex, often surprising rela- tionships between images and captions and the frequent inclusion of indirect and playful allu- sions to human experience and culture. We in- vestigate both multimodal and language-only models: the former are challenged with the car- toon images directly, while [...] formulate three tasks using over a decade 

- **dense_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Abstract and intro list the tasks; refusal counts as incorrect._
  > Lee◦ Jeff Da‡ Rowan Zellers• Robert Mankoff▲ Yejin Choi†‡ † The Allen Institute for AI ⋄ University of Utah ◦ Cornell University •OpenAI ‡ University of Washington ▲ Air Mail and Cartoon Collections jackh@allenai.org ana.marasovic@utah.edu jenah@allenai.org llee@cs.cornell.edu {jzda,rowanz}@cs.washington.edu bob@bobmankoff.com yejin@cs.washington.edu Abstract Large neural networks can now generate jokes, but do they really “understand” humor? We challenge AI models with three tasks derived from the New Yorker Cartoon Caption Con- test: matching a joke to a cartoon, identify- ing a winning caption, and explaining why a winning caption is funny. These tasks en- capsulate progressively more sop

## en-tst-21  (gold factual mistake = 1)

**Question:** Name the GPU on which the Agent modules were run. 

**Answer under evaluation:** The article does not mention the GPU used in the experiments.

- **pseudo_rag** → relevance `answers`, sufficient `True`, verdict `incorrect` — _NVIDIA-3080 12GB stated; answer claims not mentioned._
  > We ran the Agent modules on an NVIDIA-3080 GPU with 12GB memory.

- **bm25_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Window 3 states NVIDIA-3080; answer claims not mentioned._
  > due to limited comput- ing resources. Another solution is edge computing, where video data is pre-processed on edge devices and summarized for the API. While this can be a research direction, data compression may cause in- formation loss and reduce end-to-end performance. The pipeline components will need modification if developers want to adopt the framework and in- tegrate with their work. To ensure flexibility, this part should be an independent module that can run locally or in the cloud. Researchers and developers should be able to easily integrate new components into this Agent module, further challenging the sharing of large [...] want to adopt the framework and in- tegrate with their

- **dense_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _NVIDIA-3080 stated; answer claims not mentioned._
  > admin interface and permission control system to develop the bench- mark and annotation interface. Django’s maturity and large support community make it ideal for pro- duction development. The Agent module, also in Python, includes all agent related submodules, allowing deployment on suitable compute nodes without altering the architecture. Communication between the Client, API, and Agent modules will be via RESTful endpoints. For sharing large data between modules, local deployments (e.g., Client 49 on Raspberry Pi, API and Agent on local servers) will use FTP for file synchronization. In cloud so- lutions (e.g., AWS), files will be uploaded to AWS S36, [...] text-to-speech (GPT4O_ETE); a l

## en-tst-39  (gold factual mistake = 1)

**Question:** How many sentences are there in the TIMIT corpus?

**Answer under evaluation:** (The TIMIT corpus contains 1,600 sentences of 40-second duration each, and is used to study speech recognition.)The article states that the TIMIT corpus contains 1,600 sentences.

- **pseudo_rag** → relevance `answers`, sufficient `True`, verdict `incorrect` — _6300 sentences; answer says 1,600._
  > In the TIMIT corpus, there are 6300 sentences recorded by 630 speakers from eight dialect regions of the United States.

- **bm25_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _6300 sentences; answer says 1,600._
  > learn vowel and nasality features in Engish and French, two ciwGAN instances are trained sepa- rately on TIMIT Speech Corpus (Garofolo et al., 1993) and the SIWIS French Speech Synthesis Database (Yamagishi et al., 2017). The TIMIT Speech Corpus includes English raw speech sen- tences (at 16 kHz sampling rate) and their cor- responding time-aligned phonetic labels. In the TIMIT corpus, there are 6300 sentences recorded by 630 speakers from eight dialect regions of the United States. We used the entire TIMIT dataset to extract training data for the English experiment. The SIWIS French Speech Synthesis Database con- sists of [...] and their cor- responding time-aligned phonetic labels. In the 

- **dense_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _6300 sentences; answer says 1,600._
  > learn vowel and nasality features in Engish and French, two ciwGAN instances are trained sepa- rately on TIMIT Speech Corpus (Garofolo et al., 1993) and the SIWIS French Speech Synthesis Database (Yamagishi et al., 2017). The TIMIT Speech Corpus includes English raw speech sen- tences (at 16 kHz sampling rate) and their cor- responding time-aligned phonetic labels. In the TIMIT corpus, there are 6300 sentences recorded by 630 speakers from eight dialect regions of the United States. We used the entire TIMIT dataset to extract training data for the English experiment. The SIWIS French Speech Synthesis Database con- sists of [...] and their cor- responding time-aligned phonetic labels. In the 

## en-tst-472  (gold factual mistake = 1)

**Question:** Name the  three diverse classes of language understanding problems on which the evaluation focuses.

**Answer under evaluation:** According to the article "LexSym: Compositionality as Lexical Symmetry" by Akyurek, Ekin et al., the evaluation focuses on three diverse classes of language understanding problems, which are:1. **Semantic Inference**: This class involves tasks that require the model to infer the meaning of a sentence or phrase based on the context and the meanings of individual words.2. **Word Sense Disambiguation**: This class involves tasks that require the model to identify the correct meaning of a word in a given sentence or phrase, given its multiple possible meanings.3. **Natural Language Inference**: Th

- **pseudo_rag** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Paper: complex computations, derived lexicons, multi-modal data; answer fabricated._
  > Evaluation focuses on three diverse classes of language understanding problems: com-plex, context-dependent computations (Sec. 5.1), large, automatically derived lexicons (Sec. 5.2), and multi-modal data (Sec. 5.3).

- **bm25_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Windows 1-2 list the three classes; answer fabricated._
  > words or phrases (equivalent to removing cases 2 and 4). Unlike LEXSYM, these methods cannot guarantee correctness: in Fig. 1c, substitut- ing green in place of yellow yields an image with two green objects and an incorrect answer. 5 Experiments Our experiments aim to evaluate whetherLEXSYM can improve compositional generalization in down- stream models. The main goal of these experi- ments is to evaluategenerality across tasks and data modalities. Evaluation focuses on three diverse classes of language understanding problems: com- plex, context-dependent computations (Sec. 5.1), large, automatically derived lexicons (Sec. 5.2), and multi-modal data (Sec. 5.3). 5.1 Complex computations We [.

- **dense_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Windows 1-2 on formal setup; window 3 lists the three classes._
  > xn], with xi drawn from a vocabulary Σ. Finally, we assume that well-formedness can be computed by a a bi- nary interpretation function I : X → {0, 1} with I(x) = 1 iff x ∈ X. A wide variety of language understanding problems, from very simple to very complex, may be defined in this way: Example 1a: Arithmetic Language Modeling. Ex- amples x are true sentences of the form a plus b is c, where a, b and c are numbers: I(one plus two is three) = 1 but I(two plus two is five) = 0. Example 1b: Semantic [...] patches (1c). “Language understanding” in each of these tasks is encapsulated by the function I. What does it mean for I to be compositional? Under most def- initions, a compositional languag

## en-tst-317  (gold factual mistake = 1)

**Question:** What are the three components of the novel framework?

**Answer under evaluation:** According to the article "OATH-Frames: Characterizing Online Attitudes Towards Homelessness with LLM Assistants" by Ranjit, Jaspreet, et al., the three components of the novel framework are:1. **O**bjectification: This component refers to the tendency to view people experiencing homelessness as objects or entities, rather than as individuals with inherent dignity and worth.2. **A**ffect: This component captures the emotional tone of online discourse about homelessness, including feelings of empathy, anger, or indifference.3. **T**otalization: This component represents the tendency to generaliz

- **pseudo_rag** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Paper: typology discovery, annotation at scale, analysis; answer fabricated O/A/T._
  > In this work, we introduce a novel framework to understand, synthesize and characterize large-scale public attitudes towards societal issues through a collaboration between social work experts and LLMs, illustrated in Figure 1. Our framework consists of three components designed to study written discourse on social issues: (i) the discovery of a framing typology (§2), (ii) annotation of data at scale with this typology (§3), and (iii) the analysis of these collected annotations (§4). While we focus our study on homelessness, our framework is generalizable to other domains of social discourse.

- **bm25_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Window 2 lists all three components; answer fabricated O/A/T._
  > homelessness at a large scalevia online discourse has rarely been attempted in prior efforts. Our work investigates the role of LLMs in reliably assisting social work experts in characterizing such large-scale online discourse on homelessness. In this work, we introduce a novel framework to understand, synthesize and characterize large-scale public attitudes towards societal issues through a collaboration between social work experts and LLMs, illustrated in Figure 1. Our framework con- sists of three components designed to study written discourse on social issues: (i) the discovery of a framing typology (§2), (ii) annotation of data at scale with this typology (§3), [...] attitudes towards s

- **dense_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Window 1 lists all three components; answer fabricated O/A/T._
  > attitudes towards societal issues through a collaboration between social work experts and LLMs, illustrated in Figure 1. Our framework con- sists of three components designed to study written discourse on social issues: (i) the discovery of a framing typology (§2), (ii) annotation of data at scale with this typology (§3), and (iii) the analysis of these collected annotations (§4). While we fo- cus our study on homelessness, our framework is generalizable to other domains of social discourse. At the heart of our framework is our novel fram- ing typology to characterize Online Attitudes To- 2Over 653,104 U.S. individuals experienced homelessness [...] and (iii) the analysis of these collected 

## en-tst-49  (gold factual mistake = 1)

**Question:** Name the three pipelines for image transcreation.

**Answer under evaluation:** 3

- **pseudo_rag** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Names e2e-instruct, cap-edit, cap-retrieve; answer gives only '3'_
  > Contribution 2(Pipelines): In §2, we introduce three pipelines for this task – a) e2e-instruct (instruction-based image-editing): that edits images directly following a natural language instruction; b) cap-edit (caption →LLM edit →image edit): that first captions the image, makes the caption culturally relevant, and edits the original image as per the culturally-modified caption; c) cap-retrieve (caption →LLM edit →image retrieval): that uses the culturally-modified caption from cap-edit to retrieve a natural image instead. We also experiment with GPT-4o and DALLE-3 to generate new images using culturally-modified captions (§A.4).

- **bm25_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _All three names across windows; answer names none_
  > a first step towards transcreation with machine learn- ing systems, by assessing capabilities of generative models for the task of image transcreation across cultural boundaries. In text-based systems alone, models struggle with translating culture-specific information, like idioms (Liu et al., 2023). More- over, to our knowledge, automatically transcreating visual content has previously been unaddressed. Contribution 2 (Pipelines): In §2, we introduce three pipelines for this task – a) e2e-instruct (instruction-based image-editing): that edits im- ages directly following a natural language instruc- tion; b) cap-edit (caption → LLM edit → im- age edit): that first captions the image, makes t

- **dense_k3** → relevance `partial`, sufficient `True`, verdict `incorrect` — _e2e-instruct and cap-retrieve named; cap-edit name cut; answer names none_
  > caption for cultural relevance, and edits the original image using the LLM-edit as instruction; and cap-retrieve uses this LLM-edit to retrieve a natural image from a country-specific image dataset. Given the unprecedented nature of this task, we create pipelines using pre-existing SOTA models, and benchmark them on our newly created test set. some countries in the harder application dataset. 2 Pipelines for Image Transcreation We introduce three pipelines for image transcre- ation comprising of state-of-the-art generative mod- els. The code to run all pipelines with exact prompts used can be found in Table D.2. An overview of each pipeline is [...] test set. some countries in the harder app

## en-tst-212  (gold factual mistake = 1)

**Question:** How many low-resource languages does the dataset that was created cover?

**Answer under evaluation:** 2The article does not mention the number of low-resource languages covered by the dataset.

- **pseudo_rag** → relevance `answers`, sufficient `True`, verdict `incorrect` — _39 low-resource languages; answer claims not mentioned._
  > We describe a broad data collection effort in- volving around 6k professionally translated sen- tence pairs for each of 39 low-resource lan- guages, which we make publicly available.

- **bm25_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Window 1 says 39; answer claims not mentioned._
  > it does. We describe a broad data collection effort in- volving around 6k professionally translated sen- tence pairs for each of 39 low-resource lan- guages, which we make publicly available. We analyse the gains of models trained on this small but high-quality data, showing that it has significant impact even when larger but lower quality pre-existing corpora are used, or when data is augmented with millions of sentences through backtranslation. 1 Introduction State of the art machine translation models are able to cover hundreds of languages (Ma et al., 2021; Wang et al., 2022; Siddhant et al., 2022; NLLB Team [...] on individual language pairs (Tapo et al., 2021; Ali et al., 2021; Adelani

- **dense_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Window 1 says 39; window 3's 27 is a subgroup._
  > bring. With this in mind, we set up a data collec- tion effort for a number of low-resource languages which fit this criterion, resulting in a dataset of around six thousand English sentences translated into each of 39 low-resource languages. Language selection In order to choose which languages to collect data for, we took several fac- tors into account. First, we looked at the list of languages supported by Wikipedia. The user- generated encyclopedia is one of the most visited websites in the world, and constitutes an impor- tant means of knowledge dissemination for many low-resource language communities. Crucially, Wikipedia [...] languages for which not enough high quality data was cur- 

## es-tst-225  (gold factual mistake = 1)

**Question:** ¿Cómo la solución propuesta hace uso  de la resolución de referencias?

**Answer under evaluation:** En el artículo "Resolving Indirect Referring Expressions for Entity Selection" de Hosseini et al., la solución propuesta para resolver expresiones de referencia indirectas (IREs) hace uso de la resolución de referencias de varias maneras:1. **Análisis de dependencias**: La solución utiliza un análisis de dependencias para identificar las dependencias entre las expresiones y los entidades mencionadas en el texto. Esto permite determinar qué entidades están relacionadas con las expresiones de referencia indirectas.2. **Resolución de anáforas**: La solución utiliza técnicas de resolución de anáfo

- **pseudo_rag** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Candidates scored via USE/entailment/BERT/T5 with descriptions; parsing/anaphora/ontologies fabricated._
  > Indirect reference resolution can be deﬁned as follows: Given an alternative question with K choices8 C = {c1, . . . , cK}, and a RE r, models should disambiguate the choice c∗∈C intended by r. We assume r does not directly mention c∗by its name or position, but does uniquely refer to c∗. 5.1 Information Available to Models At a minimum, all models require the RE r and the names of the choices C = {c1, . . . , cK}. In addition, models may use textual descriptions {s1, . . . , sK}to aid disambiguation. We deﬁne choice text s′ i (1 ≤i ≤K) as: (a) The entity name ci, or (b) the concatenation of ci and the textual de- scription si, separated by a delimiter.9 We consider the following four experi

- **bm25_k3** → relevance `partial`, sufficient `False`, verdict `undecidable` — _Section 5.1 inputs (RE, names, descriptions); models not described._
  > and listeners reason about each others’ inten- tions and beliefs explains which attributes speakers pick to describe an entity, and how listeners disam- biguate the entity. V ogel et al. (2013); Monroe et al. (2017) focus on the pragmatic reasoning involved during the conversation which helps in reaching a common understanding of the topic. Wilkes-Gibbs and Clark (1992) study how REs change as the con- versation proceeds. In an experiment, they show that participants start from long and indeﬁnite de- scriptions of images, but end up with short and def- inite references. Jordan and Walker (2005) study the subproblem of [...] understanding of the topic. Wilkes-Gibbs and Clark (1992) study how 

- **dense_k3** → relevance `partial`, sufficient `False`, verdict `undecidable` — _Task definition only; approach not described._
  > enables for the ﬁrst time the study of how language models can be adapted to this task. We ﬁnd they achieve 82%-87% accuracy in realistic settings, which while reasonable also invites further advances. 1 Introduction Natural dialog often requires resolving referring expressions (REs), not only within and across texts, but also for grounding natural language expres- sions to speciﬁc entities or images. We focus on a speciﬁc conversational setting where a speaker’s utterance intends to disambiguate between known named entities. While many aspects of RE resolu- tion have been studied extensively, past work has focused on pragmatic reasoning (Dale and [...] a method where references to items fro

## es-tst-23  (gold factual mistake = 1)

**Question:** ¿Cuáles son los cuatro subconjuntos que componen el conjunto de datos MULTILEGALPILE?

**Answer under evaluation:** El MultiLegalPile es un corpus legal multilingüe que consta de cuatro subconjuntos principales:1. Legislación: Este subconjunto incluye leyes, actos y otros documentos legislativos de las 17 jurisdicciones cubiertas en el corpus.2. Jurisprudencia: Este subconjunto contiene decisiones judiciales y opiniones de los tribunales de las 17 jurisdicciones incluidas en el corpus.3. Tratados y Acuerdo Internacionales: Este subconjunto incluye tratados, acuerdos y convenciones internacionales relacionados con el derecho de las 17 jurisdicciones representadas en el corpus.4. Informes Jurídicos y Artículo

- **pseudo_rag** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Subsets: Native MLP, Eurlex, Legal MC4, Pile of Law; answer fabricated._
  > MULTI LEGAL PILE consists of four subsets: a) Native Multi Legal Pile (112 GB), b) Eurlex Resources (179 GB), c) Legal MC4 (106 GB) and d) Pile of Law (Henderson et al., 2022) (292 GB).

- **bm25_k3** → relevance `topical`, sufficient `False`, verdict `undecidable` — _Title, LEXTREME scores, biomedical related work; subsets not mentioned._
  > Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (V olume 1: Long Papers) , pages 15077–15094 August 11-16, 2024 ©2024 Association for Computational Linguistics MultiLegalPile: A 689GB Multilingual Legal Corpus Joel Niklaus1,2,4 Veton Matoshi2 Matthias Stürmer1,2 Ilias Chalkidis3 Daniel E. Ho4 1University of Bern 2Bern University of Applied Sciences 3University of Copenhagen 4Stanford University Abstract Large, high-quality datasets are crucial for training Large Language Models ( LLMs). However, so far, few datasets are available for specialized critical domains such as law and the available ones are often small and only in English. To fill this gap, [

- **dense_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Section 3.2 lists the four subsets; answer's legislation/jurisprudence split fabricated._
  > Spanish (Licari and Comandè, 2022; Masala et al., 2021; Gutiérrez-Fandiño et al., 2021), English remains predominant, emphasizing the need for multilingual legal corpora. Table 1 compares previous domain-specific corpora, all in English and all legal corpora less than 1/4 of MULTILEGAL PILE’s size. 3 M ULTI LEGAL PILE 3.1 Construction We transformed all datasets into xz compressed JSON Lines (JSONL) format. The combination of XZ compression and JSONL is ideal for streaming large datasets due to reduced file size and efficient decompression and reading. Filtering mC4 We used the vast multilingual web crawl corpus, mC4 (Xue et al., 2021), as [...] SPARQL queries and 3https://echanges.dila.gouv

## es-tst-352  (gold factual mistake = 1)

**Question:** ¿Cuál es el conjunto de datos de referencia de muestras de voz que se utiliza aquí?

**Answer under evaluation:** Según el artículo "SPZ: A Semantic Perturbation-based Data Augmentation Method with Zonal-Mixing for Alzheimer’s Disease Detection" de Li, FangFang et al., el conjunto de datos de referencia de muestras de voz utilizado es el "ADNI-2 dataset" (Alzheimer's Disease Neuroimaging Initiative-2 dataset).

- **pseudo_rag** → relevance `not_found`, sufficient `False`, verdict `undecidable` — _Empty context._
  > 

- **bm25_k3** → relevance `topical`, sufficient `False`, verdict `undecidable` — _Baselines and ADReSS reference; dataset use not stated._
  > AD; (2) Feature-based methods (Meghanani et al., 2021) that employ fastText (bi-grams and tri-grams) and a CNN model (bi-grams, tri-grams and 4-grams) to extract n-gram-based linguistic features; (3) BERT (Balagopalan et al., 2021) that takes the [CLS] token as the global representation for classification; (4) Fusion (Campbell et al., 2021) that integrates acoustic and linguistic features to augment the classification performance; (5) SVM (BT RU) and BERT (BT DE) (Hlédiková et al., 2022) employ Back-translation from Russian (RU) and German (DE), respectively, as the textual data augmentation for AD detection. The ensemble methods include (6) Ensemble (Sarawgi et al., 2020) [...] that integra

- **dense_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _ADReSS dataset, not ADNI-2._
  > Challenge Dataset We utilize the ADReSS dataset (Luz et al., 2020), which serves as a standardized benchmark for the Alzheimer’s Disease (AD) research community, fa- cilitating research in AD detection. This dataset comprises 156 speech samples with corresponding transcripts from participants—both non-AD (35 male, 43 female) and AD-affected (35 male, 43 female) English speakers—performing the Cookie Theft picture description task (Giles et al., 1996). Our study focuses exclusively on the transcript data, formatted according to the CHAT protocol (MacWhinney, 2000), a recognized standard for TalkBank data. As shown in Table 1, the dataset has a balanced distribution across age and [...] 43 fem

## es-tst-192  (gold factual mistake = 1)

**Question:** ¿Cuáles son las principales diferencias que los autores encuentran entre los errores humanos en comparación con las traducciones automáticas?

**Answer under evaluation:** Según el artículo "Automatic Correction of Human Translations" de Lin, Jessy, et al., los autores identifican varias diferencias importantes entre los errores humanos en traducciones y los errores en traducciones automáticas. A continuación, se presentan algunas de las principales diferencias mencionadas en el artículo:1. **Tipos de errores**: Los errores humanos suelen ser más complejos y multifactoriales, como la falta de concordancia gramatical, la inexactitud en la elección de palabras o la omisión de información importante. Por otro lado, los errores en traducciones automáticas suelen ser

- **pseudo_rag** → relevance `answers`, sufficient `True`, verdict `incorrect` — _MT: fluency errors; humans: typos, mismatches, grammar; answer's differences fabricated._
  > We annotate and release error labels for all test sentences in ASICS to enable its use as a diagnostic set for per-type evaluation of models. On the larger EMERSON and DO datasets, we randomly sample 50 errors to annotate for this analysis. Error types were annotated by a professional German translator. Each segment can have multiple error types. In Table 3, we report the percentage of sentences with at least one error of each type. 74% of sentences in APE exhibit a fluency error, in contrast to up to 22% of sentences in ACED , while other types like monolingual grammar, bilingual, and preferential errors are notably underrepresented in APE. We also note that all sentences in the APE shared 

- **bm25_k3** → relevance `partial`, sufficient `False`, verdict `undecidable` — _States errors differ and MT dominated by fluency errors; human side cut off._
  > correct errors in machine-generated translations (Simard et al., 2007). TEC is structurally identical to APE. However, it requires modeling a different data dis- tribution: errors made by humans, which differ from those made by MT systems (Freitag et al., 2021). We characterize the error distribution in TEC by building, analyzing, and releasing the ACED corpus, a collection of three TEC datasets from varying domains, with a total of 35,261 En- glish–German translations produced and corrected by professional translators in the natural course of their work. While APE is dominated by the fluency errors that are characteristic of MT systems (74% [...] appear in the train split. translation task3

- **dense_k3** → relevance `topical`, sufficient `False`, verdict `undecidable` — _Error taxonomy and claim that errors differ; differences not stated._
  > and target text, e.g. over- or under-translation, mis-translations; (3) Preferen- 2https://www.statmt.org/wmt21/ape-task.html tial edits correct text that is inconsistent with the preferences of the customer, as described in ex- tralinguistic project requirements (e.g. terminol- ogy or stylistic preferences). Examples of each error type are shown in Table 1. Our error taxon- omy closely mirrors those of previous analyses of human translation errors (Specia and Shah, 2014; Yuan and Sharoff, 2020; Gupta et al., 2021), and we confirm their findings that human translation errors differ from MT errors. However, while pre- vious work focuses on error detection and quality estimation, TEC is [...] 

## es-tst-77  (gold factual mistake = 0)

**Question:** ¿En qué aspectos son mejores los vocabularios buscados por el algoritmo VOLT que los vocabularios que se utilizan habitualmente en los entornos de traducción automática bilingüe?

**Answer under evaluation:** En el artículo "Vocabulary Learning via Optimal Transport for Neural Machine Translation" de Xu, Jingjing et al., el algoritmo VOLT busca vocabularios que sean mejores que los vocabularios utilizados habitualmente en los entornos de traducción automática bilingüe en términos de optimizar el proceso de traducción. El algoritmo VOLT utiliza una comprensión alternativa de la vocabulario desde la perspectiva de la teoría de la información para encontrar el mejor diccionario de tokens sin la necesidad de entrenamiento de prueba. Esto permite al algoritmo buscar vocabularios que sean más eficientes 

- **pseudo_rag** → relevance `answers`, sufficient `True`, verdict `correct` — _Vague; omits explicit higher BLEU and smaller vocabulary size._
  > Vocabularies Searched by VOLT are Better than Widely-used Vocabularies on Bilingual MT Settings. Ding et al. (2019) gather 42 pa- pers that have been accepted by the research track of Conference of Machine Translation (WMT) through 2017 and 2018. Among these papers, the authors ﬁnd that 30K-40K is the most popular range for the number of BPE merge actions. Following this work, we first compare our methods with dominant BPE-30K. The results are listed in Table 1. As we can see, the vocabularies searched by VOLT achieve higher BLEU scores with large size reduction. The promising results demonstrate that VOLT is a practical approach that can ﬁnd a well-performing vocabulary with higher BLEU and

- **bm25_k3** → relevance `topical`, sufficient `False`, verdict `undecidable` — _Multilingual BLEU table and appendix; bilingual comparison absent._
  > on most pairs. X-En Es Pt-br Fr Ru He Ar Ko Zh-cn It Ja Zh-tw Nl Ro BPE-60K 32.77 35.97 31.45 19.39 26.65 22.28 14.13 15.80 29.06 10.31 15.03 26.83 26.44 VOLT 33.84 37.18 32.85 20.23 26.85 22.17 14.36 16.59 30.44 10.75 15.73 27.68 27.45 X-En Tr De Vi Pl Pt Bg El Fa Sr Hu Hr Uk Cs BPE-60K 16.74 25.92 21.00 18.06 34.17 30.41 29.35 20.49 26.66 17.97 28.30 22.18 22.08 VOLT 17.55 27.01 22.25 18.93 35.64 31.77 31.27 20.05 27.45 19.00 29.25 23.34 23.54 X-En Id Th Sv Sk Sq Lt Da My Sl Mk Fr-ca Fi [...] learning for sentiment analysis: A survey. Wiley In- terdisciplinary Reviews: Data Mining and Knowl- edge Discovery, 8(4). Yi Zhao, Yanyan Shen, and Junjie Yao. 2019. Re- current neural network for te

- **dense_k3** → relevance `partial`, sufficient `False`, verdict `undecidable` — _Smaller size, competitive BLEU, speedup; answer vague but consistent._
  > in diverse scenarios. Furthermore, VOLT is a lightweight solution and does not require expen- sive computation resources. On English-German translation, VOLT only takes 30 GPU hours to ﬁnd vocabularies, while the traditional BPE-Search so- lution takes 384 GPU hours. 2 Related Work Initially, most neural models were built upon word-level vocabularies (Costa-juss `a and Fonol- losa, 2016; Vaswani et al., 2017; Zhao et al., 2019). While achieving promising results, it is a common constraint that word-level vocabularies fail on handling rare words under limited vocabu- lary sizes. Researchers recently have proposed several ad- vanced vocabularization approaches, like byte- level approaches [...

## es-tst-18  (gold factual mistake = 1)

**Question:** ¿En qué fases de la tarea compartida (shared task) participaron los autores y qué puestos obtuvieron en las competiciones? 

**Answer under evaluation:** En el artículo titulado "ApplicaAI at SemEval-2020 Task 11: On RoBERTa-CRF, Span CLS and Whether Self-Training Helps Them" escritos por Jurkiewicz, Dawid et al., los autores participaron en la tarea compartida de SemEval-2020 Task 11, que se centró en el reconocimiento de entidades nominales (NER). La tarea involucró tres sub-tareas: NER en texto plano, NER en texto de cita y NER en texto de cita con contexto.Los autores utilizaron el modelo RoBERTa-CRF para la NER y evaluaron el impacto del entrenamiento auto en el rendimiento del modelo. Los resultados de la competencia mostraron que el equi

- **pseudo_rag** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Paper: first in TC, second in SI; answer's NER subtasks fabricated._
  > This paper presents the winning system for the propaganda Technique Classiﬁcation (TC) task and the second-placed system for the propaganda Span Identiﬁcation (SI) task.

- **bm25_k3** → relevance `irrelevant`, sufficient `False`, verdict `undecidable` — _Footnote and Figure 2 token diagram; nothing on tasks or rankings._
  > re- ﬂection, whether it is a byproduct of training instability. What actually happened here was investigated afterward and will be discussed in Section 3. 1See: https://github.com/jcpeterson/openwebtext OpenWebText is a project aimed at the reconstruction of OpenAI’s unreleased WebText dataset. 1417 RoBERTaRoBERTa [BOS] Tok 1 Tok N [BOP] Tok 1' Tok M [EOP] Tok 1" Tok L... ... ... ... ... ...E[BOS] E1 EME[BOP] E1' E1" EL ... ... ...T[BOS] T1 TMT[BOP] T1' T1" TL E[EOP] T[EOP] EN TN Span label Left context Right contextSpan to classify [BOS] Tok 1 Tok N Tok 1' Tok M Tok 1" Tok L... ... [...] M [EOP] Tok 1" Tok L... ... ... ... ... ...E[BOS] E1 EME[BOP] E1' E1" EL ... ... ...T[BOS] T1 TMT[BOP] T

- **dense_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Abstract states winning TC and second-place SI; answer fabricated NER rankings._
  > Proceedings of the 14th International Workshop on Semantic Evaluation , pages 1415–1424 Barcelona, Spain (Online), December 12, 2020. 1415 ApplicaAI at SemEval-2020 Task 11: On RoBERTa-CRF, Span CLS and Whether Self-Training Helps Them Dawid Jurkiewicz∗ Łukasz Borchmann∗ Izabela Kosmala Filip Grali ´nski Applica.ai Zaj˛ ecza 15, 00-351 Warsaw, Poland firstname.lastname@applica.ai Abstract This paper presents the winning system for the propaganda Technique Classiﬁcation (TC) task and the second-placed system for the propaganda Span Identiﬁcation (SI) task. The purpose of the TC task was to identify an applied propaganda technique given propaganda text fragment. The goal of SI task was to ﬁnd 

## es-tst-37  (gold factual mistake = 1)

**Question:** ¿Cómo definen los autores la complejidad ortográfica y la complejidad fonológica de una lengua para sus experimentos?

**Answer under evaluation:** En el artículo "Language Complexity and Speech Recognition Accuracy: Orthographic Complexity Hurts, Phonological Complexity Doesn't" de Taguchi, Chihiro et al., los autores definen la complejidad ortográfica como el número de grafemas (los elementos básicos de un sistema de escritura) y la entropía unigrama de grafemas (una medida de la imprevisibilidad de la secuencia de grafemas). La complejidad fonológica, por otro lado, se mide por la logografía (la proporción de palabras que se escriben de manera fonética, es decir, representando el sonido de una palabra con caracteres individuales) y la 

- **pseudo_rag** → relevance `partial`, sufficient `True`, verdict `incorrect` — _Lists measures; logographicity is orthographic, phonological is phoneme count; answer misassigns_
  > we fine-tune the multilingual self-supervised pretrained model Wav2Vec2-XLSR-53 on 25 languages with 15 writing systems, and we compare their ASR accuracy, number of graphemes, unigram grapheme entropy, logographicity (how much word/morpheme-level information is encoded in the writing system), and number of phonemes.

- **bm25_k3** → relevance `partial`, sufficient `True`, verdict `incorrect` — _Abstract lists measures; answer assigns logographicity to phonology, invents phoneme entropy_
  > Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (V olume 1: Long Papers) , pages 15493–15503 August 11-16, 2024 ©2024 Association for Computational Linguistics Language Complexity and Speech Recognition Accuracy: Orthographic Complexity Hurts, Phonological Complexity Doesn’t Chihiro Taguchi University of Notre Dame ctaguchi@nd.edu David Chiang University of Notre Dame dchiang@nd.edu Abstract We investigate what linguistic factors affect the performance of Automatic Speech Recog- nition (ASR) models. We hypothesize that or- thographic and phonological complexities both degrade accuracy. To examine this, we fine- tune the multilingual self-supervised pre

- **dense_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Explicit: orthographic |C|,H(C),S_token; phonological |Φ|; answer wrong_
  > linguistic complexity, specifically orthographic and phonological complexity. To answer this question, this study proposes three hypotheses about factors that may make learn- ing ASR hard. Hypothesis 1. If a language has more character (grapheme) types, then ASR accuracy gets lower. This idea corresponds to the example of Chinese mentioned above. Hypothesis 2. The more a language’s writing system encodes word- or morpheme-level informa- tion, the more ASR accuracy decreases. Sproat and Gutkin (2021) call this property logographicity, as opposed to phonographicity, which means that a language’s written form is more predictable from its spoken form. Hypothesis 3. If a language [...] we were un

## es-tst-167  (gold factual mistake = 1)

**Question:** ¿Cómo proponen los autores reducir la demora o latencia en el análisis semántico conversacional estándar?

**Answer under evaluation:** Según el artículo "Online Semantic Parsing for Latency Reduction in Task-Oriented Dialogue" de Zhou et al., los autores proponen reducir la demora o latencia en el análisis semántico conversacional estándar mediante la predicción y ejecución de llamadas a funciones mientras el usuario sigue hablando. Esto se lograría mediante el enfoque de "online semantic parsing", que implica:1. **Predicción de llamadas a funciones**: El sistema predice las llamadas a funciones que se realizarán en el programa ejecutable a partir del contexto de la conversación en curso.2. **Ejecución de llamadas a funciones

- **pseudo_rag** → relevance `answers`, sufficient `True`, verdict `correct` — _Abstract: predict and execute function calls while user speaks; answer matches._
  > Standard conversational semantic parsing maps a complete user utterance into an exe- cutable program, after which the program is executed to respond to the user. This could be slow when the program contains expensive function calls. We investigate the opportunity to reduce latency by predicting and executing function calls while the user is still speaking. We introduce the task of online semantic pars- ing for this purpose, with a formal latency re- duction metric inspired by simultaneous ma- chine translation. We propose a general frame- work with ﬁrst a learned preﬁx-to-program pre- diction module, and then a simple yet effec- tive thresholding heuristic for subprogram se- lection for earl

- **bm25_k3** → relevance `topical`, sufficient `False`, verdict `undecidable` — _Program formalism and Lispress example; proposal not described._
  > a constant value, and each directed edge u 𝓁− →v∈ E represents thatv ﬁlls the𝓁 argument of the functionu. Positional arguments are given edge labels arg0, arg1, etc. We use “graph” and “program” interchangeably hereon. In task-oriented dialogue systems, an executable programG is generated in response to a user ut- terance u with possible context c from the dia- logue history. The utterance is a token sequence u = (u1,u 2,...,u |u|) and the context is also en- coded as a sequence c = (c1,c 2,...,c |c|). We use u[m] to denote themth preﬁx presented to the online system, andtm [...] the original utterance takes 0.386 seconds to 19See documentation at https://docs.microsoft. com/en-us/azure/cogn

- **dense_k3** → relevance `answers`, sufficient `True`, verdict `correct` — _Contributions and conclusion describe prefix-to-program parsing with subgraph selection._
  > Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics V olume 1: Long Papers, pages 1554 - 1576 May 22-27, 2022 c©2022 Association for Computational Linguistics Online Semantic Parsing for Latency Reduction in Task-Oriented Dialogue Jiawei Zhoum∗ Jason Eisner3 Michael Newman3 Emmanouil Antonios Platanios3 Sam Thomson3 mHarvard University 3Microsoft Semantic Machines mjzhou02@g.harvard.edu 3{jason.eisner,mike.newman, anthony.platanios,samuel.thomson}@microsoft.com Abstract Standard conversational semantic parsing maps a complete user utterance into an exe- cutable program, after which the program is executed to respond to the user. This could be slow when the

## es-tst-162  (gold factual mistake = 1)

**Question:** ¿Cuáles son los factores clave identificados como necesarios para el éxito de la NLU con modelos lingüísticos pequeños?

**Answer under evaluation:** Según el artículo "It’s Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners" de Schick, Timo et al., los factores clave identificados como necesarios para el éxito de la NLU (Natural Language Understanding) con modelos lingüísticos pequeños son:1. **Pre-training**: Los modelos lingüísticos pequeños deben ser pre-entrenados en grandes cantidades de datos para desarrollar representaciones lingüísticas útiles.2. **Data quality**: La calidad de los datos utilizados para el pre-entrenamiento es crucial para el éxito de los modelos lingüísticos pequeños.3. **Task-specific fi

- **pseudo_rag** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Factors: multiple patterns, pattern compensation, labeled data, underlying LM; answer generic._
  > We have proposed a simple yet effective modiﬁcation of PET, enabling us to use it for tasks that require predicting multiple tokens. In extensive experiments, we have identified several factors responsible for the strong performance of PET combined with ALBERT: the possibility to concurrently use multiple patterns for transforming examples into cloze questions, the ability to compensate for patterns that are difﬁcult to understand, the usage of labeled data to perform parameter updates, and the underlying LM itself.

- **bm25_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Intro lists the four factors; answer's list fabricated._
  > with 32 training examples, while requiring only 0.1% of its parameters (Fig- ure 1). Moreover, training with PET can be per- formed in several hours on a single GPU without requiring expensive hyperparameter optimization. Finally, we show that similar performance can also be achieved without unlabeled data and provide a detailed analysis of the factors contributing to PET’s strong performance: its ability to combine multiple task formulations, its resilience to word- ings that are hard to understand, its usage of la- beled data, and characteristics of the underlying LM. Given PET’s “green” properties, we see our work as an important [...] detailed analysis of the factors contributing to PET’

- **dense_k3** → relevance `partial`, sufficient `False`, verdict `undecidable` — _Abstract says factors identified but does not list them._
  > Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , pages 2339–2352 June 6–11, 2021. ©2021 Association for Computational Linguistics 2339 It’s Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners Timo Schick1,2 and Hinrich Schütze1 1 Center for Information and Language Processing, LMU Munich, Germany 2 Sulzer GmbH, Munich, Germany timo.schick@sulzer.de Abstract When scaled to hundreds of billions of pa- rameters, pretrained language models such as GPT-3 (Brown et al., 2020) achieve remark- able few-shot performance. However, enor- mous amounts of compute are required for train

## es-tst-27  (gold factual mistake = 0)

**Question:** ¿Cómo la solución propuesta hace uso  de la resolución de referencias?

**Answer under evaluation:** "Simnel cake sounds better." La solución propuesta en el artículo "Resolviendo Expresiones Refiriéndose Indirectamente para la Selección de Entidades" de Hosseini, Mohammad Javad et al. se enfoca en abordar el problema de la resolución de referencias en el contexto de la selección de entidades. La resolución de referencias es el proceso de identificar y vincular expresiones o menciones en un texto a entidades o objetos específicos en el mundo real. En el ejemplo dado, la solución tiene como objetivo identificar la entidad específica (Simnel cake o Pandan cake) a la que se refiere la expresión 

- **pseudo_rag** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Task definition fits generically, but example 'Simnel cake sounds better' is a direct mention._
  > Indirect reference resolution can be deﬁned as follows: Given an alternative question with K choices8 C = {c1, . . . , cK}, and a RE r, models should disambiguate the choice c∗∈C intended by r. We assume r does not directly mention c∗by its name or position, but does uniquely refer to c∗. 5.1 Information Available to Models At a minimum, all models require the RE r and the names of the choices C = {c1, . . . , cK}. In addition, models may use textual descriptions {s1, . . . , sK}to aid disambiguation. We deﬁne choice text s′ i (1 ≤i ≤K) as: (a) The entity name ci, or (b) the concatenation of ci and the textual de- scription si, separated by a delimiter.9 We consider the following four experi

- **bm25_k3** → relevance `partial`, sufficient `True`, verdict `incorrect` — _Model inputs and 'uniquely refers' assumption; answer's direct-mention example contradicts._
  > and listeners reason about each others’ inten- tions and beliefs explains which attributes speakers pick to describe an entity, and how listeners disam- biguate the entity. V ogel et al. (2013); Monroe et al. (2017) focus on the pragmatic reasoning involved during the conversation which helps in reaching a common understanding of the topic. Wilkes-Gibbs and Clark (1992) study how REs change as the con- versation proceeds. In an experiment, they show that participants start from long and indeﬁnite de- scriptions of images, but end up with short and def- inite references. Jordan and Walker (2005) study the subproblem of [...] understanding of the topic. Wilkes-Gibbs and Clark (1992) study how 

- **dense_k3** → relevance `answers`, sufficient `True`, verdict `incorrect` — _Indirect RE definition; answer's example names the entity directly, contradicting it._
  > enables for the ﬁrst time the study of how language models can be adapted to this task. We ﬁnd they achieve 82%-87% accuracy in realistic settings, which while reasonable also invites further advances. 1 Introduction Natural dialog often requires resolving referring expressions (REs), not only within and across texts, but also for grounding natural language expres- sions to speciﬁc entities or images. We focus on a speciﬁc conversational setting where a speaker’s utterance intends to disambiguate between known named entities. While many aspects of RE resolu- tion have been studied extensively, past work has focused on pragmatic reasoning (Dale and [...] a method where references to items fro
