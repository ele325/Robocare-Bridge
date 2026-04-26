## 13. Services Middleware complémentaires

Le middleware RoboCare ne se limite pas à la réception des données MQTT et à l’exécution du module ML. Il intègre également deux services essentiels :

- un service de notifications via Firebase Cloud Messaging ;
- un assistant intelligent basé sur l’API Groq AI ;
- une couche Firestore permettant de centraliser les données capteurs, les historiques et les réponses utilisateur.

---

## 13.1 Service de notifications — Firebase Cloud Messaging

### Rôle du service

Le service `notification_service.py` permet d’envoyer des notifications mobiles à l’utilisateur lorsqu’une situation critique ou prédictive est détectée.

Il intervient principalement dans deux cas :

| Fonction | Rôle |
|---|---|
| `send_critical_alert()` | Envoie une alerte immédiate si l’humidité est critique |
| `send_irrigation_prediction()` | Envoie une notification basée sur le score ML prédictif |

---

### 13.1.1 Alerte critique immédiate

La fonction `send_critical_alert()` est appelée lorsque l’humidité mesurée passe sous un seuil critique.

#### Étapes de fonctionnement

1. Le middleware détecte une humidité trop faible.
2. Il construit un topic utilisateur sous la forme :

```python
topic = "user_{}".format(uid)