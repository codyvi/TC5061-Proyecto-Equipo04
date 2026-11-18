from locust import HttpUser, task, between


class MiBiciUser(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def predecir_demanda_p1(self):
        payload = {
            "hora": 8,
            "dia_semana": 1,
            "es_fin_de_semana": 0,
            "es_festivo": 0,
            "es_semana_santa": 0,
            "latitud": 20.67,
            "longitud": -103.35,
            "salidas_t": 5,
            "salidas_lag_1h": 3,
            "salidas_lag_24h": 6,
            "salidas_lag_168h": 4,
            "salidas_media_24h": 2.5
        }

        with self.client.post(
            "/predict/p1",
            json=payload,
            name="POST /predict/p1",
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
            elif "pred_salidas_t1" not in response.json():
                response.failure("Respuesta sin predicción P1")

    @task(1)
    def predecir_riesgo_p2(self):
        payload = {
            "hora": 8,
            "dia_semana": 1,
            "es_fin_de_semana": 0,
            "es_festivo": 0,
            "es_semana_santa": 0,
            "latitud": 20.67,
            "longitud": -103.35,
            "capacidad": 20,
            "salidas_t": 5,
            "llegadas_t": 2,
            "salidas_lag_1h": 3,
            "salidas_lag_24h": 6,
            "salidas_lag_168h": 4,
            "llegadas_lag_1h": 2,
            "llegadas_lag_24h": 3,
            "llegadas_lag_168h": 3,
            "indice_flujo_t": -0.15,
            "indice_flujo_acum_3h": -0.3
        }

        with self.client.post(
            "/predict/p2",
            json=payload,
            name="POST /predict/p2",
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
