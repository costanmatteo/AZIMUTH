"""
Genera il documento PDF con la descrizione concettuale dell'architettura AZIMUTH.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)
from reportlab.platypus.flowables import Flowable
from pathlib import Path


# ─── Colors ───────────────────────────────────────────────────────────
BLUE_DARK = HexColor('#1a237e')
BLUE_MED = HexColor('#1565c0')
BLUE_LIGHT = HexColor('#e3f2fd')
GRAY_DARK = HexColor('#37474f')
GRAY_MED = HexColor('#78909c')
GRAY_LIGHT = HexColor('#eceff1')
GREEN = HexColor('#2e7d32')
ORANGE = HexColor('#e65100')
RED = HexColor('#c62828')


# ─── Custom Flowables ─────────────────────────────────────────────────

class BoxedNote(Flowable):
    """A colored box with text inside."""
    def __init__(self, text, bg_color=BLUE_LIGHT, border_color=BLUE_MED, width=None):
        Flowable.__init__(self)
        self.text = text
        self.bg_color = bg_color
        self.border_color = border_color
        self._width = width or 16*cm

    def wrap(self, availWidth, availHeight):
        self._width = min(self._width, availWidth)
        style = ParagraphStyle('boxnote', fontSize=9, leading=12, fontName='Helvetica')
        p = Paragraph(self.text, style)
        w, h = p.wrap(self._width - 20, availHeight)
        self._height = h + 16
        self._para = p
        return self._width, self._height

    def draw(self):
        self.canv.setFillColor(self.bg_color)
        self.canv.setStrokeColor(self.border_color)
        self.canv.setLineWidth(1.5)
        self.canv.roundRect(0, 0, self._width, self._height, 4, fill=1, stroke=1)
        self._para.drawOn(self.canv, 10, 6)


def build_pdf():
    output_path = Path(__file__).parent / "AZIMUTH_Architettura_Concettuale.pdf"

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2.5*cm, bottomMargin=2*cm,
        title="AZIMUTH - Architettura Concettuale",
        author="AZIMUTH Project"
    )

    styles = getSampleStyleSheet()

    # ── Custom styles ──
    s_title = ParagraphStyle(
        'DocTitle', parent=styles['Title'],
        fontSize=26, leading=32, textColor=BLUE_DARK,
        fontName='Helvetica-Bold', spaceAfter=6
    )
    s_subtitle = ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'],
        fontSize=13, leading=16, textColor=GRAY_MED,
        fontName='Helvetica', spaceAfter=20, alignment=TA_CENTER
    )
    s_h1 = ParagraphStyle(
        'H1', parent=styles['Heading1'],
        fontSize=18, leading=22, textColor=BLUE_DARK,
        fontName='Helvetica-Bold', spaceBefore=24, spaceAfter=10,
        borderWidth=0, borderPadding=0
    )
    s_h2 = ParagraphStyle(
        'H2', parent=styles['Heading2'],
        fontSize=14, leading=17, textColor=BLUE_MED,
        fontName='Helvetica-Bold', spaceBefore=16, spaceAfter=6
    )
    s_h3 = ParagraphStyle(
        'H3', parent=styles['Heading3'],
        fontSize=11, leading=14, textColor=GRAY_DARK,
        fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=4
    )
    s_body = ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=black,
        fontName='Helvetica', alignment=TA_JUSTIFY, spaceAfter=6
    )
    s_code = ParagraphStyle(
        'Code', parent=styles['Normal'],
        fontSize=8.5, leading=11, textColor=GRAY_DARK,
        fontName='Courier', leftIndent=12, spaceAfter=6,
        backColor=GRAY_LIGHT
    )
    s_bullet = ParagraphStyle(
        'Bullet', parent=s_body,
        leftIndent=18, bulletIndent=6,
        spaceBefore=2, spaceAfter=2
    )
    s_formula = ParagraphStyle(
        'Formula', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=BLUE_DARK,
        fontName='Courier-Bold', alignment=TA_CENTER,
        spaceBefore=6, spaceAfter=6
    )

    story = []

    def h1(text): story.append(Paragraph(text, s_h1))
    def h2(text): story.append(Paragraph(text, s_h2))
    def h3(text): story.append(Paragraph(text, s_h3))
    def p(text): story.append(Paragraph(text, s_body))
    def sp(h=6): story.append(Spacer(1, h*mm))
    def bullet(text): story.append(Paragraph(f"<bullet>&bull;</bullet> {text}", s_bullet))
    def formula(text): story.append(Paragraph(text, s_formula))
    def code(text): story.append(Paragraph(text.replace('\n', '<br/>'), s_code))
    def note(text): story.append(BoxedNote(text))
    def hr(): story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY_MED, spaceAfter=8, spaceBefore=8))

    # ══════════════════════════════════════════════════════════════════
    # TITLE PAGE
    # ══════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 60*mm))
    story.append(Paragraph("AZIMUTH", s_title))
    story.append(Paragraph("Controller Optimization System", ParagraphStyle(
        'sub2', fontSize=16, leading=20, textColor=BLUE_MED,
        fontName='Helvetica', alignment=TA_CENTER, spaceAfter=8
    )))
    story.append(Paragraph("Architettura Concettuale", s_subtitle))
    hr()
    story.append(Paragraph(
        "Documento tecnico che descrive il funzionamento concettuale del sistema "
        "di ottimizzazione dei controllori per processi industriali multi-stadio.",
        ParagraphStyle('abstract', fontSize=10, leading=14, textColor=GRAY_DARK,
                       fontName='Helvetica-Oblique', alignment=TA_CENTER, spaceAfter=20)
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS (manual)
    # ══════════════════════════════════════════════════════════════════
    h1("Indice")
    toc_items = [
        "1. Obiettivo del Sistema",
        "2. La Catena di Processi Industriali",
        "3. Modelli Causali Strutturali (SCM)",
        "4. Generazione delle Traiettorie",
        "5. Predittori di Incertezza (Fase 1)",
        "6. Il Controllore Neurale (Fase 2)",
        "7. La Process Chain: il Passaggio Forward",
        "8. La Funzione di Reliability F",
        "9. Le Funzioni di Loss",
        "10. Curriculum Learning",
        "11. Valutazione e Metriche",
        "12. Riepilogo dell'Architettura",
    ]
    for item in toc_items:
        bullet(item)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 1. OBIETTIVO
    # ══════════════════════════════════════════════════════════════════
    h1("1. Obiettivo del Sistema")

    p("AZIMUTH affronta un problema fondamentale nell'industria manifatturiera: "
      "come ottimizzare una <b>catena di processi sequenziali</b> in presenza di "
      "<b>incertezza</b> e <b>variabilita' ambientale</b>.")
    sp()

    p("Il sistema consiste in una sequenza di processi fisici (es. lavorazione laser, "
      "pulizia al plasma, deposizione galvanica, micro-incisione) dove l'output di "
      "ciascun processo influenza il successivo. Ogni processo e' soggetto a:")

    bullet("<b>Rumore di processo</b>: imprecisioni degli attuatori, rumore dei sensori, "
           "derive termiche — tutto cio' che rende il comportamento reale diverso da "
           "quello teorico.")
    bullet("<b>Condizioni strutturali</b>: parametri ambientali (temperatura, umidita') "
           "che non sono controllabili ma influenzano il risultato.")

    sp()
    p("L'obiettivo e' addestrare un <b>controllore neurale</b> che, osservando l'output "
      "(e la relativa incertezza) di ogni processo, decida gli input ottimali per il "
      "processo successivo, massimizzando una metrica di <b>reliability</b> globale F.")

    sp()
    note("<b>In sintesi:</b> Dato un ambiente incerto e variabile, il controllore impara "
         "a prendere decisioni adattive per portare l'intera catena il piu' vicino "
         "possibile al comportamento ideale.")

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 2. LA CATENA DI PROCESSI
    # ══════════════════════════════════════════════════════════════════
    h1("2. La Catena di Processi Industriali")

    p("Il sistema modella una catena di 4 processi manifatturieri, "
      "ciascuno con i propri input controllabili e non controllabili:")

    sp()
    # Process table
    proc_data = [
        ['Processo', 'Input Controllabili', 'Input Non Controllabili', 'Output'],
        ['Laser', 'PowerTarget', 'AmbientTemp', 'ActualPower'],
        ['Plasma', 'RF_Power', 'Duration', 'RemovalRate'],
        ['Galvanic', 'CurrentDensity, Duration', '(nessuno)', 'Thickness'],
        ['Microetch', 'Concentration, Duration', 'Temperature', 'RemovalDepth'],
    ]
    t = Table(proc_data, colWidths=[2.5*cm, 3.8*cm, 3.8*cm, 3*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE_DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, GRAY_MED),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, GRAY_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    sp()
    p("La distinzione tra input <b>controllabili</b> e <b>non controllabili</b> e' "
      "fondamentale: il controllore puo' decidere solo i primi, mentre i secondi "
      "rappresentano le condizioni ambientali a cui deve adattarsi.")

    sp()
    p("La catena e' <b>sequenziale e causale</b>: il controllore del processo i-esimo "
      "puo' osservare solo gli output dei processi precedenti (1, ..., i-1), "
      "non quelli futuri. Questo rispecchia la realta' fisica dove le decisioni "
      "devono essere prese in tempo reale.")

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 3. SCM
    # ══════════════════════════════════════════════════════════════════
    h1("3. Modelli Causali Strutturali (SCM)")

    p("Ogni processo fisico e' modellato da un <b>Structural Causal Model (SCM)</b>, "
      "un modello generativo che cattura le relazioni causali tra variabili e le "
      "sorgenti di incertezza.")

    sp()
    h3("Rumore Strutturale vs. Rumore di Processo")

    p("Gli SCM distinguono due tipi di variabilita':")

    bullet("<b>Rumore strutturale</b> (structural noise): variazioni nelle condizioni "
           "ambientali che creano scenari diversi. Esempio: la temperatura ambiente "
           "varia tra 10 e 40 gradi C in diverse sessioni di produzione.")
    bullet("<b>Rumore di processo</b> (process noise): imprecisioni intrinseche "
           "dell'attrezzatura. Esempio: rumore shot del laser, derive termiche, "
           "imprecisioni di misura. Questo e' sempre presente.")

    sp()
    p("Questa distinzione e' cruciale perche':")
    bullet("Il rumore strutturale definisce le <b>condizioni operative</b> (lo scenario)")
    bullet("Il rumore di processo definisce l'<b>incertezza</b> attorno al risultato atteso")
    bullet("Il controllore deve gestire entrambi: adattarsi allo scenario e "
           "compensare l'incertezza")

    sp()
    h3("Esempio: SCM del Laser")
    p("Il modello laser implementa un sistema Light-Current-Temperature (L-I-T) con:")
    bullet("Rumore lognormale moltiplicativo sull'efficienza")
    bullet("Shot noise proporzionale alla radice della potenza")
    bullet("Rumore di misura additivo")
    bullet("Deriva termica AR(1) correlata nel tempo")
    p("Il risultato e' un mapping non lineare: (PowerTarget, AmbientTemp) -> ActualPower "
      "con incertezza che dipende dal punto operativo.")

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 4. TRAIETTORIE
    # ══════════════════════════════════════════════════════════════════
    h1("4. Generazione delle Traiettorie")

    p("Il sistema genera tre tipi di traiettorie, ciascuna con un ruolo specifico:")

    sp()
    h2("4.1 Traiettoria Target (a*)")
    p("Rappresenta il <b>comportamento ideale</b>: cosa succederebbe se l'attrezzatura "
      "funzionasse perfettamente, senza rumore di processo.")

    bullet("Rumore strutturale: <b>attivo</b> (crea diversita' di scenari)")
    bullet("Rumore di processo: <b>azzerato</b> (comportamento deterministico)")
    bullet("Viene generata campionando dagli SCM con varianza di processo forzata a zero")

    sp()
    note("<b>Nota importante:</b> La traiettoria target e' un <b>array di N scenari</b>. "
         "Ogni scenario corrisponde a una diversa combinazione di condizioni ambientali. "
         "L'output ideale cambia da scenario a scenario perche' le condizioni strutturali "
         "sono diverse. Non esiste un unico target fisso: il target e' condizionato "
         "alle condizioni ambientali.")

    sp()
    h2("4.2 Traiettoria Baseline (a')")
    p("Rappresenta la <b>realta' senza controllore</b>: stessi input della target, "
      "stesse condizioni strutturali, ma con il rumore di processo attivo.")

    bullet("Input: <b>identici</b> alla target (confronto equo)")
    bullet("Condizioni strutturali: <b>identiche</b> alla target")
    bullet("Rumore di processo: <b>attivo</b> (variabilita' reale)")
    bullet("Reliability F' tipicamente bassa (0.5 - 0.7)")

    sp()
    h2("4.3 Traiettoria Attuale (a)")
    p("Prodotta durante il training dal controllore neurale. Il controllore "
      "genera input controllabili adattivi, cercando di massimizzare la reliability "
      "nonostante il rumore di processo.")

    bullet("Input controllabili: <b>generati dal controllore</b> (appresi)")
    bullet("Input non controllabili: presi dalla target (condizioni ambientali)")
    bullet("Rumore di processo: <b>attivo</b>")
    bullet("Reliability F obiettivo: avvicinarsi a F*")

    sp()
    h2("4.4 Utilizzo della Target Trajectory")
    p("La target trajectory (array di N scenari) viene usata per:")

    bullet("<b>Calcolo di F*</b>: la reliability della traiettoria ideale, calcolata "
           "dallo scenario 0 con varianza zero. E' il riferimento fisso.")
    bullet("<b>Behavioral Cloning loss</b>: durante il training, per ogni scenario, "
           "gli input generati dal controllore vengono confrontati con gli input "
           "target dello stesso scenario.")
    bullet("<b>Parametri non controllabili</b>: per ogni scenario, la ProcessChain "
           "prende i valori ambientali (es. AmbientTemp) dalla riga corrispondente "
           "della target trajectory.")
    bullet("<b>Input iniziali</b>: il primo processo della catena riceve gli input "
           "dalla target trajectory dello scenario corrente.")

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 5. PREDITTORI DI INCERTEZZA
    # ══════════════════════════════════════════════════════════════════
    h1("5. Predittori di Incertezza (Fase 1)")

    p("La prima fase dell'addestramento consiste nel creare un <b>surrogato neurale</b> "
      "per ogni processo fisico. Questi modelli imparano a predire non solo l'output "
      "atteso, ma anche la sua <b>incertezza</b>.")

    sp()
    h3("Architettura")
    p("Ogni predittore e' un MLP (Multi-Layer Perceptron) che mappa:")
    formula("input del processo  ->  (media, varianza) dell'output")

    p("La rete ha due teste di output: una per la media e una per la varianza (sempre "
      "positiva, ottenuta tramite softplus o esponenziale).")

    sp()
    h3("Funzione di Loss")
    p("L'addestramento usa la <b>Gaussian Negative Log-Likelihood</b>:")
    formula("L = -log N(y | mu, sigma^2) + alpha * ||sigma^2||^2")

    bullet("Il primo termine incoraggia previsioni accurate di media e varianza")
    bullet("Il secondo termine (penalita' sulla varianza) previene il collasso "
           "a varianza zero, che renderebbe il modello troppo sicuro di se'")

    sp()
    h3("Ruolo nel Sistema")
    p("Una volta addestrati, i predittori di incertezza vengono <b>congelati</b> "
      "(parametri non aggiornabili) e usati come componenti della ProcessChain "
      "nella Fase 2. Fungono da \"simulatori differenziabili\" dei processi fisici: "
      "dato un input, producono una distribuzione sull'output attraverso cui i "
      "gradienti possono fluire verso il controllore.")

    note("<b>Perche' servono?</b> Senza i predittori di incertezza, il controllore "
         "non potrebbe essere addestrato con gradient descent, perche' i processi "
         "fisici reali (o gli SCM) non sono differenziabili. I predittori creano "
         "un ponte differenziabile tra le decisioni del controllore e la reliability.")

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 6. IL CONTROLLORE NEURALE
    # ══════════════════════════════════════════════════════════════════
    h1("6. Il Controllore Neurale (Fase 2)")

    p("Il controllore e' composto da N-1 <b>Policy Generator</b> (uno per ogni coppia "
      "di processi consecutivi) e da un opzionale <b>Scenario Encoder</b>.")

    sp()
    h2("6.1 Policy Generator")
    p("Ogni policy generator e' un MLP che decide gli input controllabili del "
      "processo successivo, basandosi sull'output del processo precedente:")

    formula("(output_mean, output_var, params_non_controllabili, embedding)  ->  input_controllabili")

    sp()
    p("Caratteristiche chiave:")
    bullet("<b>Input</b>: media e varianza dell'output precedente (il controllore "
           "\"vede\" l'incertezza), parametri non controllabili, e un embedding "
           "opzionale dello scenario")
    bullet("<b>Output vincolato</b>: l'output passa attraverso una tanh e viene "
           "scalato nell'intervallo fisicamente ammissibile [min, max] del processo")
    bullet("<b>Causalita'</b>: il policy generator i vede solo gli output dei "
           "processi 1..i, mai quelli futuri")

    sp()
    h3("Perche' la varianza e' un input?")
    p("Se il controllore ricevesse solo la media dell'output precedente, non potrebbe "
      "distinguere tra:")
    bullet("Output = 0.5 con varianza bassa (misura affidabile)")
    bullet("Output = 0.5 con varianza alta (misura inaffidabile)")
    p("Con l'informazione sulla varianza, il controllore puo' adottare strategie "
      "diverse: piu' conservative quando l'incertezza e' alta, piu' aggressive "
      "quando e' bassa.")

    sp()
    h2("6.2 Scenario Encoder (opzionale)")
    p("Quando abilitato, lo scenario encoder mappa i parametri non controllabili "
      "(condizioni ambientali) in un embedding denso a bassa dimensionalita':")
    formula("(AmbientTemp, Duration_plasma, Temp_microetch, ...)  ->  embedding [16 dim]")
    p("Questo embedding viene concatenato all'input di ogni policy generator, "
      "permettendo al controllore di <b>adattare la sua strategia</b> alle "
      "specifiche condizioni operative.")

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 7. PROCESS CHAIN
    # ══════════════════════════════════════════════════════════════════
    h1("7. La Process Chain: il Passaggio Forward")

    p("La ProcessChain orchestra il passaggio forward dell'intera catena, "
      "collegando policy generator e predittori di incertezza in sequenza.")

    sp()
    h3("Flusso per uno scenario con 4 processi:")
    sp()

    steps = [
        ["1", "Input iniziali dal target", "Si prendono gli input del primo processo dalla target trajectory dello scenario corrente."],
        ["2", "Processo 1 (Laser)", "L'uncertainty predictor mappa gli input in (media, varianza) dell'output."],
        ["3", "Policy Generator 1", "Osserva (media, varianza) del laser + condizioni ambientali. Genera gli input controllabili per il plasma."],
        ["4", "Merge input", "Gli input controllabili (dal policy) vengono combinati con quelli non controllabili (dalla target trajectory)."],
        ["5", "Processo 2 (Plasma)", "L'uncertainty predictor mappa i nuovi input in (media, varianza) dell'output plasma."],
        ["6", "Policy Generator 2", "Osserva output plasma + condizioni. Genera input per il galvanic."],
        ["7", "... (ripeti)", "Si prosegue per galvanic e microetch."],
        ["8", "Traiettoria completa", "Si ottiene un dizionario con (input, output_mean, output_var) per ogni processo."],
    ]
    t = Table(steps, colWidths=[1*cm, 3.5*cm, 9*cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), BLUE_MED),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.3, GRAY_LIGHT),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [white, GRAY_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    sp()
    note("<b>Differenziabilita':</b> L'intera catena e' differenziabile end-to-end. "
         "I gradienti fluiscono dalla reliability F, attraverso il surrogato, "
         "i predittori di incertezza, e i policy generator fino ai pesi della rete. "
         "Questo permette di addestrare il controllore con gradient descent.")

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 8. RELIABILITY F
    # ══════════════════════════════════════════════════════════════════
    h1("8. La Funzione di Reliability F")

    p("La reliability F misura quanto bene l'intera catena di processi sta "
      "funzionando. E' un singolo scalare in [0, 1] calcolato a partire dalla "
      "traiettoria completa.")

    sp()
    h2("8.1 Quality Score per Processo")
    p("Per ogni processo, si calcola un punteggio di qualita' gaussiano:")
    formula("Q_i = exp( -(output_i - target_i)^2 / scale_i )")
    p("Dove:")
    bullet("<b>output_i</b>: l'output effettivo (o campionato) del processo i")
    bullet("<b>target_i</b>: il target adattivo per il processo i")
    bullet("<b>scale_i</b>: scala di sensibilita' (piu' piccola = piu' sensibile alle deviazioni)")

    sp()
    h2("8.2 Target Adattivi")
    p("I target non sono fissi ma si <b>adattano a cascata</b> in base agli output "
      "dei processi precedenti. Questo modella le dipendenze fisiche tra processi:")

    bullet("<b>Laser</b>: target fisso (0.8) — e' il primo processo")
    bullet("<b>Plasma</b>: target = 3.0 + 0.2 * (laser - 0.8) — se il laser e' "
           "troppo forte, il plasma deve compensare")
    bullet("<b>Galvanic</b>: target = 10.0 + 0.5 * (plasma - 5.0) + 0.4 * (laser - 0.5) "
           "— dipende da laser e plasma")
    bullet("<b>Microetch</b>: target = 20.0 + 1.5 * (laser - 0.5) + 0.3 * (plasma - 5.0) "
           "- 0.15 * (galvanic - 10.0) — dipende da tutti i processi precedenti")

    sp()
    h2("8.3 Reliability Finale")
    p("La reliability F e' la <b>media pesata</b> dei quality score:")
    formula("F = (1.0 * Q_laser + 1.0 * Q_plasma + 1.5 * Q_galvanic + 1.0 * Q_microetch) / 4.5")

    p("Il galvanic ha peso 1.5 (maggiore importanza per la qualita' del prodotto finale), "
      "gli altri processi hanno peso 1.0.")

    sp()
    h2("8.4 F* (Target Reliability)")
    p("F* rappresenta la reliability <b>ideale</b>: il valore ottenuto dalla target "
      "trajectory dello scenario 0 con varianza forzata a zero. E' un singolo scalare, "
      "calcolato una volta sola all'inizializzazione, e rappresenta l'upper bound "
      "teorico della performance.")

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 9. LOSS FUNCTIONS
    # ══════════════════════════════════════════════════════════════════
    h1("9. Le Funzioni di Loss")

    p("Il training del controllore (Fase 2) utilizza una combinazione di due loss:")

    sp()
    h2("9.1 Reliability Loss")
    formula("L_rel = scale * (F - F*)^2")

    p("Questa loss guida la reliability attuale F verso il target F*. "
      "Il fattore di scala (tipicamente 100) e' necessario perche' i valori "
      "di F sono compresi tra 0 e 1, e senza scala i gradienti sarebbero "
      "troppo piccoli per un addestramento efficace.")

    bullet("F vicino a F*: loss bassa, gradienti piccoli (fine-tuning)")
    bullet("F lontano da F*: loss alta, gradienti forti (correzione rapida)")

    sp()
    h2("9.2 Behavioral Cloning Loss")
    formula("L_BC = mean( ||input_controllore - input_target||^2 )")

    p("Questa loss incoraggia il controllore a generare input simili a quelli della "
      "traiettoria target. Ha un ruolo di <b>regolarizzazione</b>: impedisce al "
      "controllore di esplorare regioni troppo lontane dal comportamento noto.")

    bullet("Gli input sono normalizzati in [0, 1] prima del confronto")
    bullet("La loss e' mediata su tutti i processi")
    bullet("Il suo peso diminuisce durante il training (vedi curriculum learning)")

    sp()
    h2("9.3 Loss Combinata")
    formula("L_total = w_rel(epoch) * L_rel  +  lambda_BC(epoch) * L_BC")

    p("I pesi sono dinamici e cambiano durante l'addestramento secondo lo schema "
      "di curriculum learning descritto nella sezione successiva.")

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 10. CURRICULUM LEARNING
    # ══════════════════════════════════════════════════════════════════
    h1("10. Curriculum Learning")

    p("Il training utilizza una strategia di <b>curriculum learning</b> in due fasi "
      "per stabilizzare l'addestramento e migliorare la convergenza.")

    sp()
    h2("10.1 Il Problema")
    p("Senza curriculum, il controllore deve ottimizzare due obiettivi contrastanti "
      "fin dall'inizio:")
    bullet("La reliability loss richiede di massimizzare F (compito complesso)")
    bullet("La BC loss richiede di copiare la traiettoria target (compito semplice)")
    p("I gradienti iniziali sono confusi e l'addestramento puo' divergere o stagnare.")

    sp()
    h2("10.2 Fase 1: Warm-up (10% delle epoche)")

    warmup_data = [
        ['Parametro', 'Valore', 'Significato'],
        ['lambda_BC', '1.0 (alto)', 'Il controllore impara a copiare la target'],
        ['w_rel', '0.0', 'La reliability loss e\' ignorata'],
        ['Early stopping', 'Disabilitato', 'Si lascia imparare liberamente'],
    ]
    t = Table(warmup_data, colWidths=[3*cm, 3*cm, 8*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), GREEN),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, GRAY_MED),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, GRAY_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    p("Il warm-up fornisce un'<b>inizializzazione intelligente</b>: il controllore "
      "inizia gia' sapendo approssimativamente quali input generare.")

    sp()
    h2("10.3 Fase 2: Transizione Graduale (90% delle epoche)")

    cur_data = [
        ['Parametro', 'Inizio', 'Fine', 'Curva'],
        ['lambda_BC', '1.0', '0.01', 'Decadimento esponenziale'],
        ['w_rel', '0.0', '1.0', 'Curva S (sigmoide)'],
        ['Early stopping', '—', 'Attivo quando w_rel >= 0.9', '—'],
    ]
    t = Table(cur_data, colWidths=[3*cm, 2*cm, 4*cm, 4.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ORANGE),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, GRAY_MED),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, GRAY_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    sp()
    p("La transizione graduale da \"copia la target\" a \"massimizza la reliability\" "
      "permette al controllore di esplorare soluzioni migliori della target stessa, "
      "ma partendo da una base solida.")

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 11. TRAINING LOOP
    # ══════════════════════════════════════════════════════════════════
    h1("10.4 Il Training Loop")

    p("Ad ogni epoca, il controllore viene valutato su <b>tutti gli N scenari</b> "
      "con accumulo dei gradienti:")

    sp()
    steps_train = [
        ["1", "Azzeramento dei gradienti (una volta per epoca)"],
        ["2", "Shuffle casuale dell'ordine degli scenari"],
        ["3", "Per ogni scenario:"],
        ["", "   a. Forward pass attraverso la ProcessChain"],
        ["", "   b. Calcolo della reliability F"],
        ["", "   c. Calcolo della loss combinata"],
        ["", "   d. Backward pass con accumulo gradienti (loss / N_scenari)"],
        ["4", "Singolo passo dell'ottimizzatore (dopo tutti gli scenari)"],
    ]
    t = Table(steps_train, colWidths=[1*cm, 13*cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), BLUE_MED),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t)

    sp()
    p("L'accumulo dei gradienti su tutti gli scenari prima di un singolo step "
      "dell'ottimizzatore assicura che il controllore impari a soddisfare "
      "<b>tutti gli scenari simultaneamente</b>, evitando il catastrophic forgetting "
      "dove l'aggiornamento dell'ultimo scenario annullerebbe quelli precedenti.")

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 11. VALUTAZIONE
    # ══════════════════════════════════════════════════════════════════
    h1("11. Valutazione e Metriche")

    p("Al termine del training, il sistema valuta il controllore in modo approfondito:")

    sp()
    h2("11.1 Confronto delle Reliability")

    metrics_data = [
        ['Metrica', 'Significato', 'Valore Tipico'],
        ['F*', 'Reliability ideale (target, zero rumore)', '~0.95'],
        ["F'", 'Reliability baseline (senza controllore)', '~0.5 - 0.7'],
        ['F', 'Reliability con controllore addestrato', '~0.8 - 0.9'],
    ]
    t = Table(metrics_data, colWidths=[2*cm, 8*cm, 3.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE_DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (2, 0), (2, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, GRAY_MED),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, GRAY_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    sp()
    h2("11.2 Metriche Derivate")

    bullet("<b>Miglioramento</b>: (F - F') / |F'| * 100% — quanto il controllore "
           "migliora rispetto al baseline")
    bullet("<b>Target gap</b>: |F* - F| / F* * 100% — quanto il controllore "
           "e' distante dall'ideale")
    bullet("<b>Robustezza</b>: deviazione standard di F tra gli scenari — "
           "un controllore robusto ha bassa varianza")
    bullet("<b>Success rate</b>: percentuale di scenari dove F >= 0.95 * F*")
    bullet("<b>Train-test gap</b>: differenza tra F medio su train e test — "
           "indicatore di overfitting")

    sp()
    h2("11.3 Scenari di Test")
    p("Il sistema genera scenari di test <b>mai visti</b> durante l'addestramento "
      "(con seed diverso), per valutare la capacita' di generalizzazione del controllore "
      "a condizioni operative nuove.")

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # 12. RIEPILOGO ARCHITETTURA
    # ══════════════════════════════════════════════════════════════════
    h1("12. Riepilogo dell'Architettura")

    sp()
    h2("12.1 Componenti e Ruoli")

    summary_data = [
        ['Componente', 'Addestrabile?', 'Input', 'Output'],
        ['SCM Dataset', 'No', 'Random seed', '(input, output)'],
        ['Uncertainty Predictor', 'Fase 1 (poi congelato)', 'Input processo', '(media, varianza)'],
        ['Policy Generator', 'Si (Fase 2)', '(output prec., incertezza,\nambiente, embedding)', 'Input controllabili'],
        ['Scenario Encoder', 'Si (Fase 2)', 'Parametri non controllabili', 'Embedding [16 dim]'],
        ['Surrogato (ProT)', 'No (formula)', 'Traiettoria completa', 'F (scalare)'],
        ['ProcessChain', 'Orchestratore', 'batch_size, scenario_idx', 'Traiettoria completa'],
    ]
    t = Table(summary_data, colWidths=[3.2*cm, 3*cm, 4*cm, 3.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE_DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, GRAY_MED),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, GRAY_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    sp(8)
    h2("12.2 Flusso Complessivo")
    sp()

    flow_data = [
        ['Fase', 'Azione', 'Risultato'],
        ['Preparazione', 'Generazione traiettorie target e baseline\ndagli SCM', 'a* (ideale), a\' (baseline)'],
        ['Fase 1', 'Addestramento predittori di incertezza\nsu dati campionati dagli SCM', 'Modelli congelati:\ninput -> (media, var)'],
        ['Fase 2', 'Addestramento policy generator con\ncurriculum learning su N scenari', 'Controllore ottimizzato:\nF -> F*'],
        ['Valutazione', 'Test su scenari mai visti,\nconfronto F*, F\', F', 'Metriche di performance\ne report'],
    ]
    t = Table(flow_data, colWidths=[2.5*cm, 5.5*cm, 5.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE_DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 1), (0, -1), BLUE_MED),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, GRAY_MED),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, GRAY_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)

    sp(8)
    note("<b>Principio guida:</b> Il sistema separa cio' che e' fisso (SCM, predittori "
         "di incertezza, surrogato) da cio' che viene ottimizzato (policy generator, "
         "scenario encoder). Questa separazione permette un addestramento stabile e "
         "modulare, dove ogni componente ha un ruolo ben definito.")

    # ── Build PDF ──
    doc.build(story)
    print(f"PDF generato: {output_path}")
    return output_path


if __name__ == '__main__':
    build_pdf()
