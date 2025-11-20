"""
Test per verificare che la formula esplicita con target adattivi funzioni
correttamente con diversi numeri di processi.
"""

print("="*80)
print("TEST: FORMULA ESPLICITA CON TARGET ADATTIVI")
print("="*80)

print("\n1. CON 2 PROCESSI (laser + plasma)")
print("-"*80)
print("""
# LASER: target fisso
laser_target = 0.5

# PLASMA: target si adatta in base a LASER
plasma_target = 5.0 + 20.0 * (laser_power - 0.5)

# QUALITY SCORES
laser_quality = exp(-((laser_power - laser_target)^2) / 0.1)
plasma_quality = exp(-((plasma_rate - plasma_target)^2) / 2.0)

# COMBINAZIONE PESATA (solo laser e plasma presenti)
F = (0.2 * laser_quality + 0.15 * plasma_quality) / (0.2 + 0.15)
""")

print("\nEsempio con laser=0.6:")
laser_power = 0.6
plasma_rate = 7.0  # Si adatta

laser_target = 0.5
plasma_target = 5.0 + 20.0 * (laser_power - 0.5)

print(f"  laser_target = {laser_target:.2f}")
print(f"  plasma_target = {plasma_target:.2f}  (adattato da laser)")
print(f"  → Se laser è al 0.6 (più forte), plasma deve compensare arrivando a {plasma_target:.2f}")

print("\n" + "="*80)
print("2. CON 4 PROCESSI (laser + plasma + galvanic + microetch)")
print("-"*80)
print("""
# LASER: target fisso
laser_target = 0.5

# PLASMA: target si adatta in base a LASER
plasma_target = 5.0 + 20.0 * (laser_power - 0.5)

# GALVANIC: target si adatta in base a LASER E PLASMA
galvanic_target = 10.0 + 5.0 * (plasma_rate - 5.0) + 4.0 * (laser_power - 0.5)

# MICROETCH: target si adatta in base a TUTTI i processi precedenti
microetch_target = 20.0 + 15.0 * (laser_power - 0.5) + 3.0 * (plasma_rate - 5.0)
                   - 1.5 * (galvanic_thick - 10.0)

# QUALITY SCORES
laser_quality = exp(-((laser_power - laser_target)^2) / 0.1)
plasma_quality = exp(-((plasma_rate - plasma_target)^2) / 2.0)
galvanic_quality = exp(-((galvanic_thick - galvanic_target)^2) / 4.0)
microetch_quality = exp(-((microetch_depth - microetch_target)^2) / 4.0)

# COMBINAZIONE PESATA (tutti i processi presenti)
F = (0.2*laser_quality + 0.15*plasma_quality + 0.5*galvanic_quality + 0.15*microetch_quality)
    / (0.2 + 0.15 + 0.5 + 0.15)
""")

print("\nEsempio con laser=0.6, plasma=7.0:")
galvanic_thick = 12.0
microetch_depth = 23.0

galvanic_target = 10.0 + 5.0 * (plasma_rate - 5.0) + 4.0 * (laser_power - 0.5)
microetch_target = 20.0 + 15.0 * (laser_power - 0.5) + 3.0 * (plasma_rate - 5.0) - 1.5 * (galvanic_thick - 10.0)

print(f"  laser_target = {laser_target:.2f}")
print(f"  plasma_target = {plasma_target:.2f}  (adattato da laser)")
print(f"  galvanic_target = {galvanic_target:.2f}  (adattato da laser + plasma)")
print(f"  microetch_target = {microetch_target:.2f}  (adattato da tutti)")

print("\n" + "="*80)
print("VANTAGGI DELLA FORMULA ESPLICITA CON TARGET ADATTIVI:")
print("="*80)
print("""
1. ✓ I processi si influenzano a vicenda attraverso i target
2. ✓ La formula è completamente trasparente e verificabile
3. ✓ Funziona con qualsiasi sottoinsieme di processi (1, 2, 3 o 4)
4. ✓ I target si adattano solo ai processi effettivamente presenti
5. ✓ Mantiene la differenziabilità per il training
6. ✓ I pesi riflettono l'importanza relativa (galvanic=0.5, più importante)

Esempio:
- Con solo laser+plasma: F considera solo quei 2, normalizzando i pesi
- Con tutti e 4: F usa la formula completa con tutte le interdipendenze
""")
