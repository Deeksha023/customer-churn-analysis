/**
 * Customer Churn Prediction - Interactive Frontend Application
 * Handles API communication, Chart.js visualizations, form validation,
 * prediction results rendering, and local history management.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const form = document.getElementById('churn-prediction-form');
  const btnPredict = document.getElementById('btn-predict-churn');
  const spinner = document.getElementById('predict-spinner');
  const resultPlaceholder = document.getElementById('result-placeholder');
  const resultCard = document.getElementById('result-card');
  
  // Result elements
  const resRiskBadge = document.getElementById('res-risk-badge');
  const resPredictionLabel = document.getElementById('res-prediction-label');
  const resProbPercent = document.getElementById('res-prob-percent');
  const resProbBar = document.getElementById('res-prob-bar');
  const resFactorsList = document.getElementById('res-factors-list');
  const resActionsList = document.getElementById('res-actions-list');
  const resTimestamp = document.getElementById('res-timestamp');
  
  // History elements
  const historyTbody = document.getElementById('history-tbody');
  const btnClearHistory = document.getElementById('btn-clear-history');
  
  // Form input controls
  const inputTenure = document.getElementById('input-tenure');
  const tenureDisplay = document.getElementById('tenure-display');
  const inputMonthly = document.getElementById('input-monthly-charges');
  const inputTotal = document.getElementById('input-total-charges');
  const btnSyncTotal = document.getElementById('btn-sync-total');
  
  // Preset buttons
  const btnHighRisk = document.getElementById('btn-load-high-risk');
  const btnLowRisk = document.getElementById('btn-load-low-risk');
  const btnReset = document.getElementById('btn-reset-form');

  // Chart instances container
  let charts = {};

  // ---------------------------------------------------------------------------
  // 1. Navigation Highlighting
  // ---------------------------------------------------------------------------
  const navLinks = document.querySelectorAll('.nav-link');
  navLinks.forEach(link => {
    link.addEventListener('click', () => {
      navLinks.forEach(l => l.classList.remove('active'));
      link.classList.add('active');
    });
  });

  // ---------------------------------------------------------------------------
  // 2. Form Input Interactions & Synchronizations
  // ---------------------------------------------------------------------------
  // Sync tenure slider with text display and auto-update total charges
  function updateAutoTotal() {
    const tenureVal = parseFloat(inputTenure.value) || 1;
    const monthlyVal = parseFloat(inputMonthly.value) || 50;
    inputTotal.value = (tenureVal * monthlyVal).toFixed(1);
  }

  inputTenure.addEventListener('input', (e) => {
    tenureDisplay.textContent = e.target.value;
    updateAutoTotal();
  });

  inputMonthly.addEventListener('input', () => {
    updateAutoTotal();
  });

  btnSyncTotal.addEventListener('click', () => {
    updateAutoTotal();
  });

  // ---------------------------------------------------------------------------
  // 3. Preset Loaders (High Risk vs Low Risk Profiles)
  // ---------------------------------------------------------------------------
  // High-Risk Sample: Month-to-month, 2 months tenure, $95/mo, Fiber optic, Electronic check
  btnHighRisk.addEventListener('click', () => {
    setFormData({
      gender: 'Female',
      SeniorCitizen: 0,
      Partner: 'No',
      Dependents: 'No',
      tenure: 2,
      PhoneService: 'Yes',
      MultipleLines: 'No',
      InternetService: 'Fiber optic',
      OnlineSecurity: 'No',
      OnlineBackup: 'No',
      DeviceProtection: 'No',
      TechSupport: 'No',
      StreamingTV: 'No',
      StreamingMovies: 'No',
      Contract: 'Month-to-month',
      PaperlessBilling: 'Yes',
      PaymentMethod: 'Electronic check',
      MonthlyCharges: 95.0,
      TotalCharges: 190.0
    });
  });

  // Low-Risk Sample: 2-year contract, 60 months tenure, $45/mo, DSL, Tech Support
  btnLowRisk.addEventListener('click', () => {
    setFormData({
      gender: 'Male',
      SeniorCitizen: 0,
      Partner: 'Yes',
      Dependents: 'Yes',
      tenure: 60,
      PhoneService: 'Yes',
      MultipleLines: 'Yes',
      InternetService: 'DSL',
      OnlineSecurity: 'Yes',
      OnlineBackup: 'Yes',
      DeviceProtection: 'Yes',
      TechSupport: 'Yes',
      StreamingTV: 'No',
      StreamingMovies: 'No',
      Contract: 'Two year',
      PaperlessBilling: 'No',
      PaymentMethod: 'Credit card (automatic)',
      MonthlyCharges: 45.0,
      TotalCharges: 2700.0
    });
  });

  btnReset.addEventListener('click', () => {
    form.reset();
    inputTenure.value = 12;
    tenureDisplay.textContent = '12';
    inputMonthly.value = 75.0;
    updateAutoTotal();
  });

  function setFormData(data) {
    for (const [key, val] of Object.entries(data)) {
      const field = form.elements[key];
      if (field) {
        field.value = val;
      }
    }
    tenureDisplay.textContent = data.tenure;
    inputTenure.value = data.tenure;
    inputMonthly.value = data.MonthlyCharges;
    inputTotal.value = data.TotalCharges;
  }

  // ---------------------------------------------------------------------------
  // 4. Load Dataset Statistics & Charts from API
  // ---------------------------------------------------------------------------
  async function loadDashboardStats() {
    try {
      const response = await fetch('/api/stats');
      if (!response.ok) throw new Error('Failed to fetch dataset statistics');
      const data = await response.json();
      
      // Populate KPI cards
      if (data.kpis) {
        document.getElementById('val-total-customers').textContent = Number(data.kpis.total_customers).toLocaleString();
        document.getElementById('val-retained-customers').textContent = Number(data.kpis.retained_customers).toLocaleString();
        document.getElementById('val-churned-customers').textContent = Number(data.kpis.churned_customers).toLocaleString();
        document.getElementById('val-overall-churn-rate').textContent = `${data.kpis.overall_churn_rate}%`;
        const retainedRate = (100 - data.kpis.overall_churn_rate).toFixed(2);
        document.getElementById('val-retained-rate').textContent = `${retainedRate}% retention rate`;
      }

      // Render Charts
      if (data.charts) {
        renderContractChart(data.charts.contract);
        renderInternetChart(data.charts.internet);
        renderTenureChart(data.charts.tenure);
        renderPaymentChart(data.charts.payment);
      }
    } catch (err) {
      console.error('Error loading stats:', err);
    }
  }

  // Chart styling constants
  const chartColors = {
    retained: '#10b981', // Emerald green
    retainedBg: 'rgba(16, 185, 129, 0.75)',
    churned: '#ef4444',  // Coral red
    churnedBg: 'rgba(239, 68, 68, 0.85)',
    grid: 'rgba(255, 255, 255, 0.06)',
    text: '#94a3b8'
  };

  const commonChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: { color: '#f8fafc', font: { family: 'Inter', size: 12, weight: 600 } }
      },
      tooltip: {
        backgroundColor: '#1f293d',
        titleColor: '#fff',
        bodyColor: '#cbd5e1',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
        padding: 10,
        boxPadding: 4
      }
    },
    scales: {
      x: {
        grid: { color: chartColors.grid },
        ticks: { color: chartColors.text, font: { family: 'Inter', size: 11 } }
      },
      y: {
        grid: { color: chartColors.grid },
        ticks: { color: chartColors.text, font: { family: 'Inter', size: 11 } }
      }
    }
  };

  function renderContractChart(data) {
    const ctx = document.getElementById('chart-contract').getContext('2d');
    charts.contract = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: data.labels,
        datasets: [
          { label: 'Retained', data: data.retained, backgroundColor: chartColors.retainedBg, borderRadius: 6 },
          { label: 'Churned', data: data.churned, backgroundColor: chartColors.churnedBg, borderRadius: 6 }
        ]
      },
      options: commonChartOptions
    });
  }

  function renderInternetChart(data) {
    const ctx = document.getElementById('chart-internet').getContext('2d');
    charts.internet = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: data.labels,
        datasets: [
          { label: 'Retained', data: data.retained, backgroundColor: chartColors.retainedBg, borderRadius: 6 },
          { label: 'Churned', data: data.churned, backgroundColor: chartColors.churnedBg, borderRadius: 6 }
        ]
      },
      options: commonChartOptions
    });
  }

  function renderTenureChart(data) {
    const ctx = document.getElementById('chart-tenure').getContext('2d');
    charts.tenure = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: data.labels,
        datasets: [
          { label: 'Retained', data: data.retained, backgroundColor: chartColors.retainedBg, borderRadius: 6 },
          { label: 'Churned', data: data.churned, backgroundColor: chartColors.churnedBg, borderRadius: 6 }
        ]
      },
      options: commonChartOptions
    });
  }

  function renderPaymentChart(data) {
    const ctx = document.getElementById('chart-payment').getContext('2d');
    charts.payment = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: data.labels.map(l => l.replace(' (automatic)', ' (auto)')),
        datasets: [
          { label: 'Retained', data: data.retained, backgroundColor: chartColors.retainedBg, borderRadius: 6 },
          { label: 'Churned', data: data.churned, backgroundColor: chartColors.churnedBg, borderRadius: 6 }
        ]
      },
      options: commonChartOptions
    });
  }

  // ---------------------------------------------------------------------------
  // 5. Predict Churn Form Submission
  // ---------------------------------------------------------------------------
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // 1. Gather payload
    const formData = new FormData(form);
    const payload = {
      gender: formData.get('gender'),
      SeniorCitizen: parseInt(formData.get('SeniorCitizen')),
      Partner: formData.get('Partner'),
      Dependents: formData.get('Dependents'),
      tenure: parseFloat(formData.get('tenure')),
      PhoneService: formData.get('PhoneService'),
      MultipleLines: formData.get('MultipleLines'),
      InternetService: formData.get('InternetService'),
      OnlineSecurity: formData.get('OnlineSecurity'),
      OnlineBackup: formData.get('OnlineBackup'),
      DeviceProtection: formData.get('DeviceProtection'),
      TechSupport: formData.get('TechSupport'),
      StreamingTV: formData.get('StreamingTV'),
      StreamingMovies: formData.get('StreamingMovies'),
      Contract: formData.get('Contract'),
      PaperlessBilling: formData.get('PaperlessBilling'),
      PaymentMethod: formData.get('PaymentMethod'),
      MonthlyCharges: parseFloat(formData.get('MonthlyCharges')),
      TotalCharges: parseFloat(formData.get('TotalCharges'))
    };

    // 2. Validate input
    if (isNaN(payload.MonthlyCharges) || payload.MonthlyCharges <= 0) {
      alert('Please enter a valid monthly charge.');
      return;
    }
    if (isNaN(payload.tenure) || payload.tenure < 0) {
      alert('Please enter a valid tenure in months.');
      return;
    }

    // 3. UI Loading State
    btnPredict.disabled = true;
    spinner.style.display = 'inline-block';

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Inference error');
      }

      const result = await response.json();
      displayResult(result, payload);
      addToHistory(result, payload);

    } catch (err) {
      console.error('Prediction failed:', err);
      alert('Error evaluating churn prediction: ' + err.message);
    } finally {
      btnPredict.disabled = false;
      spinner.style.display = 'none';
    }
  });

  // ---------------------------------------------------------------------------
  // 6. Display Prediction Results
  // ---------------------------------------------------------------------------
  function displayResult(result, payload) {
    resultPlaceholder.style.display = 'none';
    resultCard.style.display = 'block';

    // Label and probability
    resPredictionLabel.textContent = result.prediction_label;
    resProbPercent.textContent = `${result.churn_probability_percent}%`;
    resProbBar.style.width = `${Math.min(100, Math.max(5, result.churn_probability_percent))}%`;

    // Risk badge styling
    resRiskBadge.className = 'risk-badge';
    resProbBar.className = 'prob-bar-fill';
    resFactorsList.className = 'factors-list';

    if (result.risk_level === 'High') {
      resRiskBadge.textContent = 'HIGH RISK';
      resRiskBadge.classList.add('badge-risk-high');
      resProbBar.classList.add('prob-bar-high');
    } else if (result.risk_level === 'Medium') {
      resRiskBadge.textContent = 'MEDIUM RISK';
      resRiskBadge.classList.add('badge-risk-medium');
      resProbBar.classList.add('prob-bar-medium');
    } else {
      resRiskBadge.textContent = 'LOW RISK';
      resRiskBadge.classList.add('badge-risk-low');
      resProbBar.classList.add('prob-bar-low');
      resFactorsList.classList.add('low-risk-factors');
    }

    // Populate factors
    resFactorsList.innerHTML = '';
    result.explanations.forEach(item => {
      const li = document.createElement('li');
      li.textContent = item;
      resFactorsList.appendChild(li);
    });

    // Populate recommendations
    resActionsList.innerHTML = '';
    result.recommendations.forEach(item => {
      const li = document.createElement('li');
      li.textContent = item;
      resActionsList.appendChild(li);
    });

    // Timestamp
    const now = new Date();
    resTimestamp.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    // Smooth scroll to result if on mobile
    if (window.innerWidth < 1024) {
      resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  // ---------------------------------------------------------------------------
  // 7. Prediction History Management (Session / LocalStorage)
  // ---------------------------------------------------------------------------
  const STORAGE_KEY = 'churn_prediction_history';

  function getHistory() {
    try {
      const data = localStorage.getItem(STORAGE_KEY);
      return data ? JSON.parse(data) : [];
    } catch {
      return [];
    }
  }

  function saveHistory(list) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(list.slice(0, 20))); // Keep last 20 entries
    } catch (e) {
      console.warn('Could not save history to localStorage', e);
    }
  }

  function addToHistory(result, payload) {
    const list = getHistory();
    const entry = {
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      contract: payload.Contract,
      tenure: payload.tenure,
      monthly: payload.MonthlyCharges,
      probability: result.churn_probability_percent,
      risk: result.risk_level,
      prediction: result.prediction_label
    };
    list.unshift(entry);
    saveHistory(list);
    renderHistoryTable();
  }

  function renderHistoryTable() {
    const list = getHistory();
    historyTbody.innerHTML = '';

    if (list.length === 0) {
      historyTbody.innerHTML = `
        <tr class="empty-row">
          <td colspan="6">No predictions evaluated yet. Use the prediction form above to score customer profiles.</td>
        </tr>
      `;
      return;
    }

    list.forEach(item => {
      const tr = document.createElement('tr');
      const badgeClass = item.risk === 'High' ? 'table-badge-high' : (item.risk === 'Medium' ? 'table-badge-medium' : 'table-badge-low');
      
      tr.innerHTML = `
        <td>${item.time}</td>
        <td>${item.contract} (${item.tenure} mo)</td>
        <td>$${Number(item.monthly).toFixed(2)}/mo</td>
        <td><strong>${item.probability}%</strong></td>
        <td><span class="table-badge ${badgeClass}">${item.risk}</span></td>
        <td>${item.prediction}</td>
      `;
      historyTbody.appendChild(tr);
    });
  }

  btnClearHistory.addEventListener('click', () => {
    if (confirm('Clear all local prediction history?')) {
      localStorage.removeItem(STORAGE_KEY);
      renderHistoryTable();
    }
  });

  // Initialize
  loadDashboardStats();
  renderHistoryTable();
  updateAutoTotal();
});
