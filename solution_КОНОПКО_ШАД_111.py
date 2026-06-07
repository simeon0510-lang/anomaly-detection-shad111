import pandas as pd
import numpy as np
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("АЛГОРИТМ ОБНАРУЖЕНИЯ АНОМАЛЬНОЙ АКТИВНОСТИ")
print("="*60)

# ПАРАМЕТРЫ
MAD_THRESHOLD = 4.0      # коэффициент жёсткости (выше = меньше аномалий)
MIN_SAMPLES = 5          # минимальное наблюдений для расчёта порога

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "plots").mkdir(exist_ok=True)

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================
print("\n1. Загрузка данных...")
parquet_files = glob.glob('**/*.parquet', recursive=True)
parquet_files = [f for f in parquet_files if 'invalidResp' not in f]
print(f"   Найдено файлов: {len(parquet_files)}")

dfs = []
for i, f in enumerate(parquet_files):
    if i % 5 == 0:
        print(f"   Загрузка: {i+1}/{len(parquet_files)}...")
    df_part = pd.read_parquet(f)
    dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)
print(f"   Загружено строк: {len(df):,}")

# ============================================================
# 2. ПОДГОТОВКА ДАННЫХ
# ============================================================
print("\n2. Подготовка данных...")
if 'CategoryNameDelivery' in df.columns:
    df = df.rename(columns={'CategoryNameDelivery': 'CategoryDelivery'})
df['researchdate'] = pd.to_datetime(df['researchdate'])
df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')

# ============================================================
# 3. ФИЛЬТРАЦИЯ
# ============================================================
print("\n3. Фильтрация...")
df_filtered = df[
    (df['BrandinDelivery'] == 1) & 
    (df['CategoryDelivery'].notna()) & 
    (df['CategoryDelivery'] != '') &
    (df['Weight'].notna())
].copy()
print(f"   После фильтрации: {len(df_filtered):,} строк")

# ============================================================
# 4. АГРЕГАЦИЯ ДЛЯ РАСЧЁТА daily_ots
# ============================================================
print("\n4. Агрегация...")
aggregated = df_filtered.groupby(
    ['SubjectID', 'researchdate', 'CategoryDelivery', 'BrandID', 'Brand']
).agg(
    query_count=('QueryText', 'count'),
    Weight=('Weight', 'first')
).reset_index()

aggregated['daily_ots'] = aggregated['query_count'] * aggregated['Weight']
aggregated['dayofweek'] = aggregated['researchdate'].dt.dayofweek
print(f"   Получено записей: {len(aggregated):,}")

# ============================================================
# 5. РАСЧЁТ ПОРОГОВ (медиана + MAD)
# ============================================================
print("\n5. Расчёт порогов...")
thresholds = []
for (cat, brand, dow), group in aggregated.groupby(['CategoryDelivery', 'BrandID', 'dayofweek']):
    values = group['daily_ots'].values
    if len(values) >= MIN_SAMPLES:
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad == 0:
            mad = 1
        threshold = median + MAD_THRESHOLD * mad
        thresholds.append({
            'CategoryDelivery': cat,
            'BrandID': brand,
            'dayofweek': dow,
            'threshold': threshold,
            'median': median,
            'mad': mad
        })

thresholds_df = pd.DataFrame(thresholds)
print(f"   Рассчитано порогов: {len(thresholds_df)}")

# ============================================================
# 6. ПОИСК АНОМАЛИЙ
# ============================================================
print("\n6. Поиск аномалий...")
aggregated = aggregated.merge(thresholds_df, on=['CategoryDelivery', 'BrandID', 'dayofweek'], how='left')
anomalies_mask = (aggregated['daily_ots'] > aggregated['threshold']) & (aggregated['threshold'].notna())
anomaly_triggers = aggregated[anomalies_mask].copy()
print(f"   Найдено аномалий: {len(anomaly_triggers)}")

# ============================================================
# 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================
print("\n7. Сохранение результатов...")

# 7.1. Уникальные пары для удаления
anomalies_to_remove = anomaly_triggers[['SubjectID', 'researchdate']].drop_duplicates()
anomalies_to_remove.to_csv(OUTPUT_DIR / 'anomalies.csv', index=False)
print(f"   anomalies.csv: {len(anomalies_to_remove)} пар")

# 7.2. Причины аномалий (с полным объяснением)
anomaly_reasons = anomaly_triggers[[
    'SubjectID', 'researchdate', 'BrandID', 'Brand', 'CategoryDelivery',
    'daily_ots', 'threshold', 'median', 'mad'
]].copy()
anomaly_reasons['score'] = anomaly_reasons['daily_ots']
anomaly_reasons['reason'] = anomaly_reasons.apply(
    lambda r: f"daily_ots ({r['daily_ots']:.1f}) > порог ({r['threshold']:.1f}) "
              f"[медиана={r['median']:.1f}, MAD={r['mad']:.1f}] для {r['CategoryDelivery']}/{r['Brand']}",
    axis=1
)
anomaly_reasons.to_csv(OUTPUT_DIR / 'anomaly_reasons.csv', index=False, encoding='utf-8-sig')
print(f"   anomaly_reasons.csv: {len(anomaly_reasons)} записей")

# ============================================================
# 8. ОСНОВНЫЕ ГРАФИКИ (обязательные по заданию)
# ============================================================
print("\n8. Построение обязательных графиков...")

# График 1: Количество аномалий по дням
daily_counts = anomaly_triggers.groupby('researchdate').size()
plt.figure(figsize=(12, 5))
plt.bar(daily_counts.index, daily_counts.values, color='red', alpha=0.7, edgecolor='darkred')
plt.xlabel('Дата')
plt.ylabel('Количество аномалий')
plt.title('Ежедневное количество аномальных активностей')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plots' / 'daily_anomaly_count.png', dpi=150)
plt.close()

# График 2: OTS до и после удаления
ots_before = aggregated.groupby('researchdate')['daily_ots'].sum()
normal_users = aggregated[~anomalies_mask][['SubjectID', 'researchdate']].drop_duplicates()
df_clean_for_plots = aggregated.merge(
    normal_users.assign(is_normal=True), 
    on=['SubjectID', 'researchdate'], 
    how='inner'
)
ots_after = df_clean_for_plots.groupby('researchdate')['daily_ots'].sum()

plt.figure(figsize=(12, 5))
plt.plot(ots_before.index, ots_before.values, 'b-o', label='До удаления', linewidth=2)
plt.plot(ots_after.index, ots_after.values, 'g-s', label='После удаления', linewidth=2)
plt.xlabel('Дата')
plt.ylabel('Общий OTS')
plt.title('Изменение общего OTS после удаления аномалий')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plots' / 'total_ots_before_after.png', dpi=150)
plt.close()

# График 3: Изменение OTS по категориям
category_before = aggregated.groupby('CategoryDelivery')['daily_ots'].sum()
category_after = df_clean_for_plots.groupby('CategoryDelivery')['daily_ots'].sum()
category_change = (category_after / category_before - 1) * 100
category_change = category_change.sort_values()

plt.figure(figsize=(10, 8))
colors = ['red' if x < 0 else 'green' for x in category_change.values]
plt.barh(category_change.index, category_change.values, color=colors, alpha=0.7, edgecolor='black')
plt.xlabel('Изменение OTS (%)')
plt.title('Изменение OTS по категориям после удаления аномалий')
plt.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plots' / 'category_ots_change.png', dpi=150)
plt.close()

print(f"   Графики сохранены в {OUTPUT_DIR}/plots/")

# ============================================================
# 9. АНАЛИТИЧЕСКИЕ ВОЗМОЖНОСТИ (по требованию задания)
# ============================================================
print("\n9. Построение аналитических графиков...")

# 9.1. Сохраняем "чистый" датафрейм для аналитики
df_clean_original = df[~df['SubjectID'].isin(anomalies_to_remove['SubjectID'])]

# 9.2. График по полу
gender_before = df_filtered.groupby('Пол').apply(
    lambda g: (g.groupby('SubjectID')['Weight'].first() * g.groupby('SubjectID').size()).sum()
)
gender_after = df_filtered[~df_filtered['SubjectID'].isin(anomalies_to_remove['SubjectID'])].groupby('Пол').apply(
    lambda g: (g.groupby('SubjectID')['Weight'].first() * g.groupby('SubjectID').size()).sum()
)
gender_change = (gender_after / gender_before - 1) * 100

plt.figure(figsize=(8, 5))
colors = ['red' if x < 0 else 'green' for x in gender_change.values]
plt.bar(gender_change.index, gender_change.values, color=colors, alpha=0.7, edgecolor='black')
plt.xlabel('Пол')
plt.ylabel('Изменение OTS (%)')
plt.title('Изменение OTS по полу после удаления аномалий')
plt.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plots' / 'gender_ots_change.png', dpi=150)
plt.close()
print("   gender_ots_change.png")

# 9.3. График по возрасту
age_before = df_filtered.groupby('Возраст').apply(
    lambda g: (g.groupby('SubjectID')['Weight'].first() * g.groupby('SubjectID').size()).sum()
)
age_after = df_filtered[~df_filtered['SubjectID'].isin(anomalies_to_remove['SubjectID'])].groupby('Возраст').apply(
    lambda g: (g.groupby('SubjectID')['Weight'].first() * g.groupby('SubjectID').size()).sum()
)
age_change = (age_after / age_before - 1) * 100
age_change = age_change.sort_values()

plt.figure(figsize=(10, 5))
colors = ['red' if x < 0 else 'green' for x in age_change.values]
plt.bar(age_change.index, age_change.values, color=colors, alpha=0.7, edgecolor='black')
plt.xlabel('Возраст')
plt.ylabel('Изменение OTS (%)')
plt.title('Изменение OTS по возрасту после удаления аномалий')
plt.xticks(rotation=45)
plt.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plots' / 'age_ots_change.png', dpi=150)
plt.close()
print("   age_ots_change.png")

# 9.4. График по типу ресурса
resource_before = df_filtered.groupby('ResourceType').apply(
    lambda g: (g.groupby('SubjectID')['Weight'].first() * g.groupby('SubjectID').size()).sum()
)
resource_after = df_filtered[~df_filtered['SubjectID'].isin(anomalies_to_remove['SubjectID'])].groupby('ResourceType').apply(
    lambda g: (g.groupby('SubjectID')['Weight'].first() * g.groupby('SubjectID').size()).sum()
)
resource_change = (resource_after / resource_before - 1) * 100

plt.figure(figsize=(10, 5))
colors = ['red' if x < 0 else 'green' for x in resource_change.values]
plt.bar(resource_change.index, resource_change.values, color=colors, alpha=0.7, edgecolor='black')
plt.xlabel('Тип ресурса')
plt.ylabel('Изменение OTS (%)')
plt.title('Изменение OTS по типу ресурса после удаления аномалий')
plt.xticks(rotation=45)
plt.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plots' / 'resource_ots_change.png', dpi=150)
plt.close()
print("   resource_ots_change.png")

# 9.5. График по платформе
platform_before = df_filtered.groupby('Platform').apply(
    lambda g: (g.groupby('SubjectID')['Weight'].first() * g.groupby('SubjectID').size()).sum()
)
platform_after = df_filtered[~df_filtered['SubjectID'].isin(anomalies_to_remove['SubjectID'])].groupby('Platform').apply(
    lambda g: (g.groupby('SubjectID')['Weight'].first() * g.groupby('SubjectID').size()).sum()
)
platform_change = (platform_after / platform_before - 1) * 100

plt.figure(figsize=(6, 5))
colors = ['red' if x < 0 else 'green' for x in platform_change.values]
plt.bar(platform_change.index, platform_change.values, color=colors, alpha=0.7, edgecolor='black')
plt.xlabel('Платформа')
plt.ylabel('Изменение OTS (%)')
plt.title('Изменение OTS по платформе после удаления аномалий')
plt.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plots' / 'platform_ots_change.png', dpi=150)
plt.close()
print("   platform_ots_change.png")

# 9.6. Таблица поисковых запросов для аномального пользователя
if len(anomaly_triggers) > 0:
    sample_anomaly = anomaly_triggers.iloc[0]
    sample_user = sample_anomaly['SubjectID']
    sample_date = sample_anomaly['researchdate']
    
    user_queries = df[
        (df['SubjectID'] == sample_user) & 
        (df['researchdate'] == sample_date)
    ][['QueryText', 'Brand', 'CategoryDelivery', 'ResourceName']]
    
    if len(user_queries) > 0:
        user_queries.to_csv(OUTPUT_DIR / f'queries_anomaly_{sample_user}_{sample_date.date()}.csv', index=False, encoding='utf-8-sig')
        print(f"   queries_anomaly_{sample_user}_{sample_date.date()}.csv")
        
        # Вывод примера запросов в консоль
        print(f"\n   Пример запросов аномального пользователя {sample_user} за {sample_date.date()}:")
        for _, row in user_queries.head(5).iterrows():
            query_preview = row['QueryText'][:60] + '...' if len(row['QueryText']) > 60 else row['QueryText']
            print(f"     - {query_preview} (бренд: {row['Brand']})")

# 9.7. График OTS по дням для топ-бренда
if len(anomaly_triggers) > 0:
    top_brand = anomaly_triggers['Brand'].value_counts().index[0]
    brand_before = df_filtered[df_filtered['Brand'] == top_brand].groupby('researchdate').apply(
        lambda g: (g.groupby('SubjectID')['Weight'].first() * g.groupby('SubjectID').size()).sum()
    )
    brand_after = df_filtered[
        (df_filtered['Brand'] == top_brand) & 
        (~df_filtered['SubjectID'].isin(anomalies_to_remove['SubjectID']))
    ].groupby('researchdate').apply(
        lambda g: (g.groupby('SubjectID')['Weight'].first() * g.groupby('SubjectID').size()).sum()
    )
    
    if len(brand_before) > 0:
        plt.figure(figsize=(12, 5))
        plt.plot(brand_before.index, brand_before.values, 'b-o', label='До удаления', linewidth=2)
        plt.plot(brand_after.index, brand_after.values, 'g-s', label='После удаления', linewidth=2)
        plt.xlabel('Дата')
        plt.ylabel('OTS')
        plt.title(f'Изменение OTS по дням для бренда: {top_brand}')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'plots' / f'brand_{top_brand}_trend.png', dpi=150)
        plt.close()
        print(f"   brand_{top_brand}_trend.png")

# ============================================================
# 10. ВЫВОД ИТОГОВОЙ СТАТИСТИКИ
# ============================================================
print("\n" + "="*60)
print("ИТОГОВАЯ СТАТИСТИКА")
print("="*60)
total_user_days = aggregated[['SubjectID','researchdate']].drop_duplicates().shape[0]
anomaly_count = len(anomalies_to_remove)
print(f"Уникальных пользователь-дней: {total_user_days:,}")
print(f"Аномальных пользователь-дней: {anomaly_count:,}")
print(f"Доля аномалий: {anomaly_count/total_user_days*100:.2f}%")
print(f"Потеря OTS: {(1 - ots_after.sum()/ots_before.sum())*100:.2f}%")
print("="*60)

print("\n" + "="*60)
print("ВЫПОЛНЕНИЕ ЗАВЕРШЕНО")
print("="*60)
print("\nПапка output содержит:")
print("  - anomalies.csv (список для удаления)")
print("  - anomaly_reasons.csv (причины аномалий)")
print("  - plots/ (все графики)")