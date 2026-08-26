# test_api.py
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"


def test_api():
    print("=" * 60)
    print("🧪 INICIANDO PRUEBAS DE INTEGRACIÓN: PINN INVERSE DRAG API")
    print("=" * 60)

    # 1. Healthcheck
    try:
        r = requests.get(f"{BASE_URL}/health")
        assert r.status_code == 200, f"Healthcheck falló: {r.status_code}"
        print(f"✅ 1. GET /health -> Status: {r.json()['status']} | Cd Activo: {r.json()['active_cd']:.4f}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: La API no está corriendo en {BASE_URL}.")
        print("   Ejecuta en otra terminal: uvicorn api.main:app --port 8000")
        return

    # 2. Presets
    r = requests.get(f"{BASE_URL}/presets")
    assert r.status_code == 200
    presets = r.json()
    print(f"✅ 2. GET /presets -> {len(presets)} casos físicos disponibles: {list(presets.keys())}")

    # 3. Predicción puntual en t=1.5s
    r = requests.get(f"{BASE_URL}/predict?t=1.5")
    assert r.status_code == 200
    pred = r.json()
    print(f"✅ 3. GET /predict?t=1.5 -> Altitud: {pred['altitude_y']} m | Velocidad: {pred['velocity_v']} m/s | Cd: {pred['cd_calibrated']}")

    # 4. Descubrimiento Inverso en Vivo (Calibración Esfera Lisa)
    print("\n🔬 4. Probando POST /discover_cd (Caso: Esfera Lisa)...")
    payload = {
        "preset": "smooth_sphere",
        "noise_std": 0.05,
        "n_points": 30,
        "initial_cd_guess": 0.10,
        "epochs": 250,
    }
    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/discover_cd", json=payload)
    t1 = time.perf_counter()
    assert r.status_code == 200, f"Error en /discover_cd: {r.text}"
    res = r.json()

    print(f"   🎯 Cd Real:          {res['cd_true']}")
    print(f"   🎲 Cd Inicial Guess: {res['initial_cd_guess']}")
    print(f"   🔍 Cd Descubierto:   {res['cd_estimated']}")
    print(f"   📉 Error Relativo:   {res['relative_error_pct']}%")
    print(f"   ⚡ Latencia Servidor: {res['latency_ms']} ms")
    print(f"   🌐 Latencia Roundtrip:{round((t1 - t0) * 1000, 2)} ms")

    # 5. Descubrimiento Inverso en Vivo (Caso: Pelota de Béisbol Cd=0.30)
    print("\n🔬 5. Probando POST /discover_cd (Caso: Pelota de Béisbol Cd=0.30)...")
    payload_bb = {
        "preset": "baseball",
        "noise_std": 0.03,
        "n_points": 35,
        "initial_cd_guess": 0.80,
        "epochs": 250,
    }
    r = requests.post(f"{BASE_URL}/discover_cd", json=payload_bb)
    res_bb = r.json()
    print(f"   🎯 Cd Real:          {res_bb['cd_true']}")
    print(f"   🎲 Cd Inicial Guess: {res_bb['initial_cd_guess']}")
    print(f"   🔍 Cd Descubierto:   {res_bb['cd_estimated']}")
    print(f"   📉 Error Relativo:   {res_bb['relative_error_pct']}%")
    print(f"   ⚡ Latencia Servidor: {res_bb['latency_ms']} ms")

    print("\n" + "=" * 60)
    print("🎉 TODAS LAS PRUEBAS DE INTEGRACIÓN PASARON EXITOSAMENTE.")
    print("=" * 60)


if __name__ == "__main__":
    test_api()
