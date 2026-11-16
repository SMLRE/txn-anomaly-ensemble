-- =====================================================================
-- 01  Transaction-level features, each in the context of its own client
-- ---------------------------------------------------------------------
-- An amount is only unusual relative to *this* client's normal, so most
-- features are standardised within client.
-- =====================================================================
WITH base AS (
    SELECT *,
           LN(amount)                                          AS log_amount,
           HOUR(ts) + MINUTE(ts) / 60.0                        AS hour_of_day,
           CASE WHEN DAYOFWEEK(ts) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
           ROW_NUMBER() OVER (PARTITION BY client_id, counterparty_id ORDER BY ts) AS cpty_seq,
           MIN(ts) OVER (PARTITION BY client_id)               AS client_first_ts
    FROM transactions
),
client_stats AS (
    SELECT client_id,
           MEDIAN(log_amount)                  AS med_log_amt,
           QUANTILE_CONT(log_amount, 0.75) - QUANTILE_CONT(log_amount, 0.25) AS iqr_log_amt,
           AVG(hour_of_day)                    AS mean_hour,
           STDDEV_SAMP(hour_of_day)            AS sd_hour
    FROM base GROUP BY client_id
)
SELECT b.txn_id, b.client_id, b.ts, b.amount, b.is_anomaly, b.anomaly_type,
       b.log_amount,
       (b.log_amount - s.med_log_amt) / NULLIF(s.iqr_log_amt / 1.349, 0)               AS amount_robust_z,
       ABS(b.hour_of_day - s.mean_hour) / NULLIF(s.sd_hour, 0)                         AS hour_z,
       b.is_weekend,
       CASE WHEN b.cpty_seq = 1 AND b.ts > b.client_first_ts + INTERVAL 14 DAY THEN 1 ELSE 0 END AS new_counterparty,
       COUNT(*) OVER (PARTITION BY b.client_id ORDER BY b.ts
                      RANGE BETWEEN INTERVAL 60 MINUTE PRECEDING AND CURRENT ROW) - 1   AS txns_prev_60min,
       CASE WHEN (b.amount BETWEEN 4750000 AND 4999999.99) OR (b.amount BETWEEN 47500000 AND 49999999.99)
            THEN 1 ELSE 0 END                                                           AS near_threshold
FROM base b JOIN client_stats s USING (client_id)
ORDER BY b.txn_id
