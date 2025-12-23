// RewardScope Dashboard - Chart.js Configuration and WebSocket Handler

// Chart.js default configuration
Chart.defaults.color = '#999';
Chart.defaults.borderColor = '#333';
Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';

// Reward timeline chart
const rewardCtx = document.getElementById('reward-chart').getContext('2d');
const rewardChart = new Chart(rewardCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Reward',
            data: [],
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 5,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            intersect: false,
            mode: 'index',
        },
        scales: {
            x: { 
                display: false,
                grid: { display: false }
            },
            y: { 
                beginAtZero: false,
                grid: { color: '#222' }
            }
        },
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: '#1a1a1a',
                borderColor: '#333',
                borderWidth: 1,
            }
        },
        animation: {
            duration: 300
        }
    }
});

// Component breakdown chart
const componentCtx = document.getElementById('component-chart').getContext('2d');
const componentChart = new Chart(componentCtx, {
    type: 'doughnut',
    data: {
        labels: [],
        datasets: [{
            data: [],
            backgroundColor: [
                '#3b82f6', '#10b981', '#f59e0b', 
                '#ef4444', '#8b5cf6', '#ec4899',
                '#06b6d4', '#84cc16', '#f97316'
            ],
            borderWidth: 0,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { 
                position: 'right',
                labels: {
                    padding: 15,
                    usePointStyle: true,
                    pointStyle: 'circle',
                }
            },
            tooltip: {
                backgroundColor: '#1a1a1a',
                borderColor: '#333',
                borderWidth: 1,
                callbacks: {
                    label: function(context) {
                        const label = context.label || '';
                        const value = context.parsed || 0;
                        const total = context.dataset.data.reduce((a, b) => a + b, 0);
                        const percentage = ((value / total) * 100).toFixed(1);
                        return `${label}: ${value.toFixed(2)} (${percentage}%)`;
                    }
                }
            }
        }
    }
});

// Episode history chart
const episodeCtx = document.getElementById('episode-chart').getContext('2d');
const episodeChart = new Chart(episodeCtx, {
    type: 'bar',
    data: {
        labels: [],
        datasets: [{
            label: 'Episode Reward',
            data: [],
            backgroundColor: '#10b981',
            borderRadius: 4,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { 
                display: true,
                grid: { display: false },
                title: {
                    display: true,
                    text: 'Episode',
                    color: '#666'
                }
            },
            y: { 
                beginAtZero: false,
                grid: { color: '#222' },
                title: {
                    display: true,
                    text: 'Total Reward',
                    color: '#666'
                }
            }
        },
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: '#1a1a1a',
                borderColor: '#333',
                borderWidth: 1,
            }
        }
    }
});

// WebSocket connection for live updates
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(`${protocol}//${window.location.host}/ws/live`);

ws.onopen = () => {
    console.log('WebSocket connected');
    document.getElementById('connection-status').textContent = 'Connected';
    document.getElementById('connection-status').classList.add('connected');
};

ws.onclose = () => {
    console.log('WebSocket disconnected');
    document.getElementById('connection-status').textContent = 'Disconnected';
    document.getElementById('connection-status').classList.remove('connected');
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'step_update') {
        // Update live stats
        document.getElementById('current-step').textContent = data.step;
        document.getElementById('current-episode').textContent = data.episode;
        
        // Calculate hacking score from components if available
        if (data.components && Object.keys(data.components).length > 0) {
            const componentValues = Object.values(data.components);
            const maxComponent = Math.max(...componentValues.map(Math.abs));
            const totalReward = Math.abs(data.reward);
            const hackingScore = totalReward > 0 ? (maxComponent / totalReward) : 0;
            document.getElementById('hacking-score').textContent = hackingScore.toFixed(2);
        }
        
        // Update reward chart
        rewardChart.data.labels.push(data.step);
        rewardChart.data.datasets[0].data.push(data.reward);
        
        // Keep last 100 points
        if (rewardChart.data.labels.length > 100) {
            rewardChart.data.labels.shift();
            rewardChart.data.datasets[0].data.shift();
        }
        
        rewardChart.update('none'); // No animation for performance
    }
};

// Initial data fetch
async function fetchInitialData() {
    try {
        // Reward history
        const rewardRes = await fetch('/api/reward-history?n=100');
        const rewardData = await rewardRes.json();
        if (!rewardData.error && rewardData.steps) {
            rewardChart.data.labels = rewardData.steps;
            rewardChart.data.datasets[0].data = rewardData.rewards;
            rewardChart.update();
            
            // Update live stats if we have data
            if (rewardData.steps.length > 0) {
                const lastIdx = rewardData.steps.length - 1;
                document.getElementById('current-step').textContent = rewardData.steps[lastIdx];
                document.getElementById('current-episode').textContent = rewardData.episodes[lastIdx];
            }
        }
        
        // Component breakdown
        const componentRes = await fetch('/api/component-breakdown?n=100');
        const componentData = await componentRes.json();
        if (!componentData.error && componentData.components) {
            componentChart.data.labels = componentData.components;
            componentChart.data.datasets[0].data = componentData.values;
            componentChart.update();
        }
        
        // Episode history
        const episodeRes = await fetch('/api/episode-history?n=50');
        const episodeData = await episodeRes.json();
        if (!episodeData.error && episodeData.episodes) {
            episodeChart.data.labels = episodeData.episodes.map(e => `Ep ${e}`);
            episodeChart.data.datasets[0].data = episodeData.total_rewards;
            episodeChart.update();

            // Check for live hacking score first (in-progress episode)
            const liveRes = await fetch('/api/live-hacking');
            const liveData = await liveRes.json();

            if (!liveData.error && liveData.in_progress) {
                // Show live score with indicator
                const scoreText = `${liveData.current_score.toFixed(2)} (in progress)`;
                document.getElementById('hacking-score').textContent = scoreText;
            } else {
                // Fall back to latest completed episode score
                if (episodeData.hacking_scores && episodeData.hacking_scores.length > 0) {
                    const latestScore = episodeData.hacking_scores[episodeData.hacking_scores.length - 1];
                    document.getElementById('hacking-score').textContent = latestScore.toFixed(2);
                }
            }
        }
    } catch (error) {
        console.error('Error fetching initial data:', error);
    }
}

// Fetch initial data on page load
fetchInitialData();

// Refresh charts periodically (every 5 seconds)
setInterval(async () => {
    try {
        // Update component breakdown
        const componentRes = await fetch('/api/component-breakdown?n=100');
        const componentData = await componentRes.json();
        if (!componentData.error && componentData.components) {
            componentChart.data.labels = componentData.components;
            componentChart.data.datasets[0].data = componentData.values;
            componentChart.update();
        }

        // Update episode history and hacking score
        const episodeRes = await fetch('/api/episode-history?n=50');
        const episodeData = await episodeRes.json();
        if (!episodeData.error && episodeData.episodes) {
            episodeChart.data.labels = episodeData.episodes.map(e => `Ep ${e}`);
            episodeChart.data.datasets[0].data = episodeData.total_rewards;
            episodeChart.update();

            // Check for live hacking score first (in-progress episode)
            const liveRes = await fetch('/api/live-hacking');
            const liveData = await liveRes.json();

            if (!liveData.error && liveData.in_progress) {
                // Show live score with indicator
                const scoreText = `${liveData.current_score.toFixed(2)} (in progress)`;
                document.getElementById('hacking-score').textContent = scoreText;
            } else {
                // Fall back to latest completed episode score
                if (episodeData.hacking_scores && episodeData.hacking_scores.length > 0) {
                    const latestScore = episodeData.hacking_scores[episodeData.hacking_scores.length - 1];
                    document.getElementById('hacking-score').textContent = latestScore.toFixed(2);
                }
            }
        }
    } catch (error) {
        console.error('Error updating charts:', error);
    }
}, 5000);

// HTMX event handlers for alerts
document.body.addEventListener('htmx:afterSwap', function(event) {
    if (event.detail.target.id === 'alerts-list') {
        // Parse the alerts and render them
        const alertsData = event.detail.xhr.response;
        try {
            const data = JSON.parse(alertsData);
            if (data.alert_groups && data.alert_groups.length > 0) {
                let html = '';
                data.alert_groups.forEach((group, index) => {
                    const severityClass = group.max_severity > 0.7 ? 'alert-item' :
                                         group.max_severity > 0.4 ? 'alert-item warning' :
                                         'alert-item info';
                    const groupId = `alert-group-${index}`;

                    html += `
                        <div class="${severityClass} alert-group" data-group-id="${groupId}">
                            <div class="alert-header" onclick="toggleAlertGroup('${groupId}')">
                                <div class="alert-main">
                                    <div class="type">${group.description}</div>
                                    <span class="count-badge">${group.count}x</span>
                                </div>
                                <div class="alert-meta">
                                    <span class="episode-info">Episode ${group.episode}</span>
                                    <span class="severity-info">Severity: ${(group.max_severity * 100).toFixed(0)}%</span>
                                    <span class="chevron">▼</span>
                                </div>
                            </div>
                            <div class="alert-details" id="${groupId}" style="display: none;">
                                <div class="occurrences-container">
                                    <p class="occurrences-note">
                                        This alert was detected ${group.count} time${group.count > 1 ? 's' : ''} in episode ${group.episode}.
                                    </p>
                                    <p class="occurrences-info">
                                        <em>Note: Individual timestamps are not currently tracked. Future versions will show detailed occurrence timestamps.</em>
                                    </p>
                                </div>
                            </div>
                        </div>
                    `;
                });
                event.detail.target.innerHTML = html;
            } else {
                event.detail.target.innerHTML = '<p class="no-alerts">No alerts detected ✓</p>';
            }
        } catch (e) {
            console.error('Error parsing alerts:', e);
        }
    }
});

// Toggle alert group expansion
function toggleAlertGroup(groupId) {
    const detailsDiv = document.getElementById(groupId);
    const groupDiv = document.querySelector(`[data-group-id="${groupId}"]`);
    const chevron = groupDiv.querySelector('.chevron');

    if (detailsDiv.style.display === 'none') {
        detailsDiv.style.display = 'block';
        chevron.textContent = '▲';
        groupDiv.classList.add('expanded');
    } else {
        detailsDiv.style.display = 'none';
        chevron.textContent = '▼';
        groupDiv.classList.remove('expanded');
    }
}

