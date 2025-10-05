<?php
/**
 * MarkovAssetModel
 * ----------------
 * Cadeia de Markov para estados de retorno (Down/Flat/Up) com simulação de cenários.
 *
 * - Discretização por thresholds (ex.: -1% / +1%).
 * - Matriz de transição com suavização de Laplace.
 * - Simulação com bootstrap de retornos condicionais ao estado sorteado.
 * - Sumário com probabilidade de perda, VaR (preço), quantis e etc.
 *
 * Obs.: PHP não é otimizado para numérico pesado; para milhares de cenários/horizontes longos,
 * considere reduzir n_scenarios ou migrar o núcleo numérico para uma lib C/Ext ou Python.
 */
class MarkovAssetModel {
    // Estados fixos
    const STATE_DOWN = 0;
    const STATE_FLAT = 1;
    const STATE_UP = 2;

    private array $returns = []; // retornos diários (float[])
    private array $states = []; // estados discretizados (int[])
    private array $retBuckets = []; // map estado -> float[] (retornos condicionais)
    private array $P = []; // matriz de transição 3x3 (linhas somam 1)
    private float $downTh; // threshold Down
    private float $upTh; // threshold Up
    private float $alpha; // suavização de Laplace
    private ?float $lastPrice = null; // último preço observado (se tiver preços)

    public function __construct(float $downThreshold = -0.01, float $upThreshold = 0.01, float $laplaceAlpha = 1.0) {
        if ($downThreshold >= $upThreshold) {
            throw new InvalidArgumentException("downThreshold deve ser < upThreshold.");
        }
        $this->downTh = $downThreshold;
        $this->upTh   = $upThreshold;
        $this->alpha  = $laplaceAlpha;
    }

    /** Fit a partir de preços (Close). Retornos simples: r_t = P_t/P_{t-1} - 1. */
    public function fitFromPrices(array $prices): void {
        if (count($prices) < 3) {
            throw new InvalidArgumentException("Forneça pelo menos 3 preços para calcular retornos/estados.");
        }
        $this->returns = [];
        for ($i = 1; $i < count($prices); $i++) {
            $p0 = (float) $prices[$i - 1];
            $p1 = (float) $prices[$i];
            if ($p0 == 0.0) {
                throw new InvalidArgumentException("Preço zero encontrado no índice " . ($i - 1));
            }
            $this->returns[] = $p1 / $p0 - 1.0;
        }
        $this->lastPrice = (float) end($prices);
        $this->discretizeAndEstimate();
    }

    /** Fit a partir de retornos já calculados. (Se quiser simular preço, informe startPrice depois.) */
    public function fitFromReturns(array $returns): void {
        if (count($returns) < 2) {
            throw new InvalidArgumentException("Forneça ao menos 2 retornos.");
        }
        $this->returns = array_map('floatval', $returns);
        $this->lastPrice = null; // desconhecido por enquanto
        $this->discretizeAndEstimate();
    }

    /** Define explicitamente o último preço, útil quando fitFromReturns foi usado. */
    public function setLastPrice(float $price): void {
        if ($price <= 0) {
            throw new InvalidArgumentException("Preço precisa ser positivo.");
        }
        $this->lastPrice = $price;
    }

    /** Retorna a matriz de transição 3x3 (array de arrays). */
    public function getTransitionMatrix(): array {
        return $this->P;
    }

    /** Retorna contadores de estados condicionais (para inspeção). */
    public function getStateHistogram(): array {
        $h = [0, 0, 0];
        foreach ($this->states as $s) {
            $h[$s]++;
        }
        return $h;
    }

    /**
     * Simula caminhos de preço por H dias úteis.
     * @param int $horizon Dias à frente.
     * @param int $nScenarios Número de cenários (linhas).
     * @param float|null $startPrice Se null, usa $lastPrice (se houver) ou 100.0.
     * @return array [paths, endStates] onde paths: float[nScenarios][horizon+1] endStates: int[nScenarios]
     */
    public function simulate(int $horizon, int $nScenarios, ?float $startPrice = null): array {
        if ($horizon <= 0 || $nScenarios <= 0) {
            throw new InvalidArgumentException("Parâmetros inválidos: horizon e nScenarios devem ser > 0.");
        }
        if ($startPrice === null) {
            $startPrice = $this->lastPrice ?? 100.0;
        }
        // Estado inicial = último estado observado
        $startState = end($this->states);
        if ($startState === false) {
            throw new RuntimeException("Modelo não calibrado. Rode fitFromPrices/fitFromReturns primeiro.");
        }

        // Pré-calcula acumuladas das linhas de P para amostragem rápida
        $cumP = [];
        for ($i = 0; $i < 3; $i++) {
            $cum = [];
            $acc = 0.0;
            for ($j = 0; $j < 3; $j++) {
                $acc += $this->P[$i][$j];
                $cum[] = $acc;
            }
            // normaliza por segurança
            $last = end($cum);
            if (abs($last - 1.0) > 1e-12) {
                for ($j = 0; $j < 3; $j++) {
                    $cum[$j] /= $last;
                }

            }
            $cumP[$i] = $cum;
        }

        $paths     = [];
        $endStates = [];
        for ($s = 0; $s < $nScenarios; $s++) {
            $p = $startPrice;
            $st = $startState;
            $path = array_fill(0, $horizon + 1, 0.0);
            $path[0] = $p;

            for ($t = 1; $t <= $horizon; $t++) {
                $st = $this->sampleNextState($st, $cumP);
                $r = $this->sampleReturnFromState($st);
                $p *= (1.0 + $r);
                $path[$t] = $p;
            }
            $paths[]     = $path;
            $endStates[] = $st;
        }
        return [$paths, $endStates];
    }

    /**
     * Sumário de risco/retorno a partir dos caminhos simulados.
     * @param array $paths float[nScenarios][horizon+1]
     * @param float $startPrice preço inicial usado na simulação (para retorno final)
     * @return array associativo com métricas
     */
    public function summarize(array $paths, float $startPrice): array {
        $finals = [];
        foreach ($paths as $row) {
            $finals[] = (float) end($row);
        }
        $rets = array_map(fn($pf) => $pf / $startPrice - 1.0, $finals);

        sort($finals);
        sort($rets);

        $probLoss = $this->mean(array_map(fn($r) => $r < 0 ? 1.0 : 0.0, $rets));
        $var5Price = $this->quantile($finals, 0.05);
        $p95 = $this->quantile($finals, 0.95);

        return [
            // Preço Inicial
            "start_price" => $startPrice,
            // Preço Médio
            "median_price" => $this->quantile($finals, 0.50),
            // Probabilidade de retorno negativo (chance de perda)
            "prob_neg_return" => $probLoss,
            // VaR 5% (Preço) (nível de preço no pior 5% dos cenários)
            "VaR5_price" => $var5Price,
            // Preço no percentil 95 (cenários otimistas)
            "p95_price" => $p95,
            // Retorno médio
            "mean_return" => $this->mean($rets),
            // Desvio-padrão do retorno (volatilidade dos retornos finais)
            "std_return" => $this->std($rets),
        ];
    }

    // ========================= Internals =========================

    private function discretizeAndEstimate(): void {
        // 1) Discretização em estados
        $this->states = [];
        foreach ($this->returns as $r) {
            if ($r < $this->downTh) {
                $this->states[] = self::STATE_DOWN;
            } elseif ($r > $this->upTh) {
                $this->states[] = self::STATE_UP;
            } else {
                $this->states[] = self::STATE_FLAT;
            }
        }
        if (count($this->states) < 2) {
            throw new RuntimeException("Poucos estados para estimar transições.");
        }

        // 2) Contagem de transições
        $counts = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        for ($t = 0; $t < count($this->states) - 1; $t++) {
            $i = $this->states[$t];
            $j = $this->states[$t + 1];
            $counts[$i][$j] += 1;
        }

        // 3) Matriz de transição com Laplace
        $P = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        for ($i = 0; $i < 3; $i++) {
            $rowSum = array_sum($counts[$i]) + 3 * $this->alpha;
            for ($j = 0; $j < 3; $j++) {
                $P[$i][$j] = ($counts[$i][$j] + $this->alpha) / $rowSum;
            }
        }
        $this->P = $P;

        // 4) Baldes de retornos por estado (para bootstrap)
        $this->retBuckets = [
            self::STATE_DOWN => [],
            self::STATE_FLAT => [],
            self::STATE_UP   => [],
        ];
        for ($t = 0; $t < count($this->returns); $t++) {
            $this->retBuckets[$this->states[$t]][] = $this->returns[$t];
        }
    }

    /** Sorteia próximo estado dado o atual, usando a cumulativa de P. */
    private function sampleNextState(int $currentState, array $cumP): int {
        $u = mt_rand() / mt_getrandmax(); // U(0,1)
        $cum = $cumP[$currentState];
        if ($u <= $cum[0]) {
            return 0;
        }
        if ($u <= $cum[1]) {
            return 1;
        }
        return 2;
    }

    /** Sorteia um retorno (bootstrap) condicional ao estado escolhido. */
    private function sampleReturnFromState(int $state): float {
        $bucket = $this->retBuckets[$state] ?? [];
        if (empty($bucket)) {
            // fallback: usa toda a amostra se o balde estiver vazio
            $bucket = $this->returns;
        }
        $idx = mt_rand(0, count($bucket) - 1);
        return (float) $bucket[$idx];
    }

    // ========================= Helpers =========================

    private function mean(array $x): float {
        $n = count($x);
        if ($n === 0) {
            return NAN;
        }
        return array_sum($x) / $n;
    }

    private function std(array $x): float {
        $n = count($x);
        if ($n < 2) {
            return NAN;
        }
        $m = $this->mean($x);
        $acc = 0.0;
        foreach ($x as $v) {
            $d = $v - $m;
            $acc += $d * $d;
        }
        return sqrt($acc / ($n - 1));
    }

    /** Quantil empírico p em [0,1], com interpolação linear entre vizinhos. */
    private function quantile(array $sorted, float $p): float {
        if ($p <= 0) {
            return (float) $sorted[0];
        }
        $n = count($sorted);
        if ($p >= 1) {
            return (float) $sorted[$n - 1];
        }
        // vetor deve estar ordenado
        $idx = ($n - 1) * $p;
        $i = (int) floor($idx);
        $f = $idx - $i;
        if ($i + 1 >= $n) {
            return (float) $sorted[$i];
        }
        return (1 - $f) * (float) $sorted[$i] + $f * (float) $sorted[$i + 1];
    }
}
?>
