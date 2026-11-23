from src.statistics.occupancy_store import OccupancyStore


class OccupancyAssistant:
    def __init__(self, store=None):
        self.store = store or OccupancyStore()

    def ask(self, message: str) -> str:
        command = message.lower().strip()
        summary = self.store.summary()
        if any(word in command for word in ("ocupacion", "cuantas", "cuantos")):
            return (
                f"Ocupacion maxima registrada: {summary['ocupacion_maxima']}. "
                f"Entradas: {summary['entradas']}. Salidas: {summary['salidas']}."
            )
        if any(word in command for word in ("entrada", "entraron")):
            return f"Entradas registradas: {summary['entradas']}."
        if any(word in command for word in ("salida", "salieron")):
            return f"Salidas registradas: {summary['salidas']}."
        if "ayuda" in command:
            return "Pregunte por ocupacion, entradas o salidas."
        return "Comando no reconocido. Escriba ayuda para ver opciones."

