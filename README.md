# Gaussian Spectra

Приложение для построения и сопоставления теоретических ИК/Raman-спектров Gaussian с экспериментальными двухколоночными спектрами.

Technical documentation is maintained in [docs/README.md](docs/README.md).

## Run

```bash
python3 -m pip install -r requirements.txt
python3 spectra_app.py
```

## Возможности

- Импорт завершённых Gaussian `.log/.out`, включая частоты, ИК-интенсивности, Raman activity и коэффициенты деполяризации.
- Gaussian, Lorentzian и Voigt-уширение; масштаб частот, нормализация, диапазон и направление оси.
- Относительное Raman-представление с настройками лазера, температуры и поляризации.
- Импорт `.csv/.txt/.xy` и baseline-коррекция ALS, airPLS, линейным и полиномиальным методами.

Подробный сценарий работы приведён в [руководстве пользователя](docs/user-guide.md).
