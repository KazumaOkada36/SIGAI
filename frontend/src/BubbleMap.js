import React, { useEffect, useRef, useState } from 'react';
import { Info, Maximize2, Download } from 'lucide-react';
import './BubbleMap.css';

function BubbleMap({ insights, adType }) {
  const canvasRef = useRef(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const animationRef = useRef(null);
  const nodesRef = useRef([]);
  const mouseRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    if (!insights) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Parse insights into bubble nodes
    const nodes = createNodesFromInsights(insights, adType);
    nodesRef.current = nodes;

    let animationTime = 0;

    const animate = () => {
      animationTime += 0.01;
      
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Update and draw nodes
      updateNodes(nodes, canvas, animationTime);
      drawConnections(ctx, nodes, canvas);
      drawNodes(ctx, nodes, canvas, hoveredNode);

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [insights, hoveredNode]);

  const createNodesFromInsights = (insights, type) => {
    const nodes = [];
    const centerX = 400;
    const centerY = 300;

    // Central node
    nodes.push({
      id: 'center',
      label: 'Your Ad',
      value: insights.summary?.overall_score || 7,
      x: centerX,
      y: centerY,
      targetX: centerX,
      targetY: centerY,
      radius: 60,
      color: '#5b6ff8',
      category: 'main',
      description: insights.summary?.description || 'Advertisement Analysis',
      score: insights.summary?.overall_score
    });

    // Create nodes from features
    let nodeId = 0;
    const categories = type === 'image' 
      ? ['engagement_predictors', 'emotional_signals', 'visual_composition', 'predicted_performance']
      : ['overall_engagement', 'video_dynamics', 'predicted_performance'];

    categories.forEach((category, catIndex) => {
      const categoryData = type === 'image' 
        ? insights.features?.[category]
        : insights.features?.video_level_features?.[category];

      if (!categoryData) return;

      const angle = (catIndex / categories.length) * Math.PI * 2;
      const distance = 200;
      const categoryX = centerX + Math.cos(angle) * distance;
      const categoryY = centerY + Math.sin(angle) * distance;

      // Category hub node
      const categoryScore = calculateCategoryScore(categoryData);
      nodes.push({
        id: `cat-${catIndex}`,
        label: formatLabel(category),
        value: categoryScore,
        x: categoryX,
        y: categoryY,
        targetX: categoryX,
        targetY: categoryY,
        radius: 45,
        color: getCategoryColor(catIndex),
        category: 'hub',
        description: getDescriptionForCategory(category),
        score: categoryScore,
        parent: 'center'
      });

      // Individual metric nodes
      Object.entries(categoryData).forEach(([key, value], index) => {
        if (typeof value === 'number' && value >= 0 && value <= 10) {
          const subAngle = angle + (index - Object.keys(categoryData).length / 2) * 0.3;
          const subDistance = 120;
          const nodeX = categoryX + Math.cos(subAngle) * subDistance;
          const nodeY = categoryY + Math.sin(subAngle) * subDistance;

          nodes.push({
            id: `node-${nodeId++}`,
            label: formatLabel(key),
            value: value,
            x: nodeX,
            y: nodeY,
            targetX: nodeX,
            targetY: nodeY,
            radius: 15 + (value / 10) * 15,
            color: getScoreColor(value),
            category: 'metric',
            description: `${formatLabel(key)}: ${value}/10`,
            score: value,
            parent: `cat-${catIndex}`
          });
        } else if (typeof value === 'boolean') {
          const subAngle = angle + (index - Object.keys(categoryData).length / 2) * 0.3;
          const subDistance = 120;
          const nodeX = categoryX + Math.cos(subAngle) * subDistance;
          const nodeY = categoryY + Math.sin(subAngle) * subDistance;

          nodes.push({
            id: `node-${nodeId++}`,
            label: formatLabel(key),
            value: value ? 8 : 3,
            x: nodeX,
            y: nodeY,
            targetX: nodeX,
            targetY: nodeY,
            radius: 20,
            color: value ? '#10b981' : '#ef4444',
            category: 'boolean',
            description: `${formatLabel(key)}: ${value ? 'Yes' : 'No'}`,
            score: value ? 'Yes' : 'No',
            parent: `cat-${catIndex}`
          });
        }
      });
    });

    return nodes;
  };

  const updateNodes = (nodes, canvas, time) => {
    const rect = canvas.getBoundingClientRect();
    const mouse = mouseRef.current;

    nodes.forEach((node, i) => {
      // Gentle floating animation
      const floatSpeed = 0.5 + (i % 3) * 0.2;
      const floatAmount = 5;
      node.y = node.targetY + Math.sin(time * floatSpeed + i) * floatAmount;
      node.x = node.targetX + Math.cos(time * floatSpeed * 0.7 + i) * floatAmount * 0.5;

      // Mouse interaction
      const dx = mouse.x - node.x;
      const dy = mouse.y - node.y;
      const distance = Math.sqrt(dx * dx + dy * dy);

      if (distance < 100) {
        const force = (100 - distance) / 100;
        node.x -= (dx / distance) * force * 10;
        node.y -= (dy / distance) * force * 10;
      }

      // Boundary check
      node.x = Math.max(node.radius, Math.min(rect.width - node.radius, node.x));
      node.y = Math.max(node.radius, Math.min(rect.height - node.radius, node.y));
    });
  };

  const drawConnections = (ctx, nodes, canvas) => {
    ctx.strokeStyle = 'rgba(203, 213, 224, 0.3)';
    ctx.lineWidth = 2;

    nodes.forEach(node => {
      if (node.parent) {
        const parent = nodes.find(n => n.id === node.parent);
        if (parent) {
          ctx.beginPath();
          ctx.moveTo(parent.x, parent.y);
          ctx.lineTo(node.x, node.y);
          ctx.stroke();
        }
      }
    });
  };

  const drawNodes = (ctx, nodes, canvas, hoveredNode) => {
    nodes.forEach(node => {
      const isHovered = hoveredNode?.id === node.id;
      const scale = isHovered ? 1.1 : 1;

      // Shadow for depth
      ctx.shadowColor = 'rgba(0, 0, 0, 0.2)';
      ctx.shadowBlur = 15;
      ctx.shadowOffsetX = 5;
      ctx.shadowOffsetY = 5;

      // Outer glow for hovered nodes
      if (isHovered) {
        ctx.shadowColor = node.color;
        ctx.shadowBlur = 30;
      }

      // Draw circle
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.radius * scale, 0, Math.PI * 2);
      
      // Gradient fill
      const gradient = ctx.createRadialGradient(
        node.x - node.radius * 0.3,
        node.y - node.radius * 0.3,
        0,
        node.x,
        node.y,
        node.radius
      );
      gradient.addColorStop(0, lightenColor(node.color, 20));
      gradient.addColorStop(1, node.color);
      
      ctx.fillStyle = gradient;
      ctx.fill();

      // Border
      ctx.strokeStyle = 'white';
      ctx.lineWidth = isHovered ? 4 : 2;
      ctx.stroke();

      // Reset shadow
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;

      // Draw text
      ctx.fillStyle = 'white';
      ctx.font = `bold ${node.category === 'main' ? 14 : node.category === 'hub' ? 11 : 9}px -apple-system, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      if (node.category === 'main' || node.category === 'hub') {
        ctx.fillText(node.label, node.x, node.y - 5);
        ctx.font = `${node.category === 'main' ? 20 : 16}px -apple-system, sans-serif`;
        ctx.fillText(typeof node.score === 'number' ? `${node.score}/10` : node.score, node.x, node.y + 10);
      } else if (node.radius > 20 || isHovered) {
        ctx.fillText(typeof node.score === 'number' ? `${node.score}` : node.score, node.x, node.y);
      }
    });
  };

  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    mouseRef.current = { x, y };

    // Check for hover
    const hoveredNode = nodesRef.current.find(node => {
      const dx = x - node.x;
      const dy = y - node.y;
      return Math.sqrt(dx * dx + dy * dy) < node.radius;
    });

    setHoveredNode(hoveredNode || null);
    canvas.style.cursor = hoveredNode ? 'pointer' : 'default';
  };

  const handleClick = (e) => {
    if (hoveredNode) {
      setSelectedNode(hoveredNode);
    }
  };

  const exportAsImage = () => {
    const canvas = canvasRef.current;
    const url = canvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = 'ad-insights-bubble-map.png';
    link.href = url;
    link.click();
  };

  return (
    <div className="bubble-map-container">
      <div className="bubble-map-header">
        <div>
          <h3>
            <Info size={20} />
            Interactive Insights Map
          </h3>
          <p>Hover over bubbles to explore metrics • Larger bubbles = better performance</p>
        </div>
        <div className="bubble-map-actions">
          <button className="map-action-btn" onClick={exportAsImage}>
            <Download size={18} />
            Export
          </button>
        </div>
      </div>

      <div className="bubble-map-canvas-wrapper">
        <canvas
          ref={canvasRef}
          className="bubble-map-canvas"
          onMouseMove={handleMouseMove}
          onClick={handleClick}
          onMouseLeave={() => setHoveredNode(null)}
        />

        {hoveredNode && (
          <div className="bubble-tooltip" style={{
            left: hoveredNode.x + 10,
            top: hoveredNode.y - 30
          }}>
            <div className="tooltip-title">{hoveredNode.label}</div>
            <div className="tooltip-description">{hoveredNode.description}</div>
          </div>
        )}
      </div>

      {selectedNode && (
        <div className="node-detail-panel">
          <div className="detail-header">
            <div className="detail-title">{selectedNode.label}</div>
            <button className="close-detail" onClick={() => setSelectedNode(null)}>×</button>
          </div>
          <div className="detail-content">
            <div className="detail-score" style={{ color: selectedNode.color }}>
              {typeof selectedNode.score === 'number' ? `${selectedNode.score}/10` : selectedNode.score}
            </div>
            <p>{selectedNode.description}</p>
            {typeof selectedNode.score === 'number' && (
              <div className="detail-recommendation">
                {getRecommendation(selectedNode.label, selectedNode.score)}
              </div>
            )}
          </div>
        </div>
      )}

      <div className="bubble-legend">
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#5b6ff8' }}></div>
          <span>Main Metrics</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#10b981' }}></div>
          <span>Strong (7-10)</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#f59e0b' }}></div>
          <span>Medium (4-6)</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#ef4444' }}></div>
          <span>Weak (1-3)</span>
        </div>
      </div>
    </div>
  );
}

// Helper functions
function calculateCategoryScore(data) {
  const numbers = Object.values(data).filter(v => typeof v === 'number' && v >= 0 && v <= 10);
  if (numbers.length === 0) return 5;
  return Math.round(numbers.reduce((a, b) => a + b, 0) / numbers.length * 10) / 10;
}

function getScoreColor(score) {
  if (score >= 7) return '#10b981';
  if (score >= 4) return '#f59e0b';
  return '#ef4444';
}

function getCategoryColor(index) {
  const colors = ['#5b6ff8', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b'];
  return colors[index % colors.length];
}

function formatLabel(str) {
  return str.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function lightenColor(color, percent) {
  const num = parseInt(color.replace('#', ''), 16);
  const amt = Math.round(2.55 * percent);
  const R = (num >> 16) + amt;
  const G = (num >> 8 & 0x00FF) + amt;
  const B = (num & 0x0000FF) + amt;
  return '#' + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
    (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
    (B < 255 ? B < 1 ? 0 : B : 255))
    .toString(16).slice(1);
}

function getDescriptionForCategory(category) {
  const descriptions = {
    'engagement_predictors': 'Metrics predicting user engagement',
    'emotional_signals': 'Emotional impact assessment',
    'visual_composition': 'Visual design quality',
    'predicted_performance': 'Expected campaign performance',
    'overall_engagement': 'Video engagement metrics',
    'video_dynamics': 'Video pacing and flow'
  };
  return descriptions[category] || category;
}

function getRecommendation(label, score) {
  if (score >= 7) {
    return `✅ Excellent ${label.toLowerCase()}! This is a strong point of your ad.`;
  } else if (score >= 4) {
    return `⚠️ ${label} is moderate. Consider enhancing this aspect for better results.`;
  } else {
    return `❌ ${label} needs improvement. This could significantly impact performance.`;
  }
}

export default BubbleMap;