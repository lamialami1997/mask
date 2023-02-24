import numpy as np
import matplotlib.pyplot as plt

# Charger les données depuis le fichier
data = np.genfromtxt('./mask_16.dat', delimiter=';')

# Créer un graphique à deux axes
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# Définir le titre et les étiquettes des axes
ax1.set_title('Comparaison des versions de GCC et ICX')
ax1.set_xlabel('Versions')
ax1.set_ylabel('Temps écoulé (ns)')
ax1.set_ylim([0, 15000000])
ax2.set_ylabel('Débit (GB/s)')
ax2.set_ylim([0, 70])
ax2.set_yticks(np.arange(0, 71, 10))

# Tracer les données
ax1.plot(data[:,0], data[:,2], color='red', marker='o', linestyle='-', label='GCC')
#ax1.plot(data[:,0], data[:,6], color='blue', marker='o', linestyle='-', label='ICX')
#ax2.plot(data[:,0], data[:,3], color='red', marker='s', linestyle='--', label='Débit GCC')
#ax2.plot(data[:,0], data[:,7], color='blue', marker='s', linestyle='--', label='Débit ICX')

# Ajouter une légende
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

# Afficher le graphique
plt.show()
