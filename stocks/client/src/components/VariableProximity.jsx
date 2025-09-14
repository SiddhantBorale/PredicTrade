import React, { useEffect, useMemo, useRef, useState } from 'react';

/**
 * VariableProximity
 * - Splits text into spans and morphs each letter's font weight / optical size
 *   based on cursor proximity.
 *
 * Props:
 *  - text: string (required)
 *  - radius: px distance where effect reaches full strength (default 120)
 *  - falloff: 'linear' | 'exponential' | 'gaussian' (default 'linear')
 *  - weightFrom / weightTo: 100..1000 (default 400 → 900)
 *  - opszFrom / opszTo: optical size (default 14 → 72)
 *  - minOpacity / maxOpacity: 0..1 (default 0.6 → 1)
 *  - className: extra classes for the wrapper
 */
export default function VariableProximity({
  text,
  radius = 120,
  falloff = 'linear',
  weightFrom = 400,
  weightTo = 900,
  opszFrom = 14,
  opszTo = 72,
  minOpacity = 0.6,
  maxOpacity = 1,
  className = '',
}) {
  const containerRef = useRef(null);
  const spanRefs = useRef([]);
  const [mouse, setMouse] = useState({ x: null, y: null });
  const letters = useMemo(() => Array.from(text || ''), [text]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const handleMove = (e) => {
      const rect = el.getBoundingClientRect();
      setMouse({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    };
    const handleLeave = () => setMouse({ x: null, y: null });

    el.addEventListener('mousemove', handleMove);
    el.addEventListener('mouseleave', handleLeave);
    // touch
    el.addEventListener('touchmove', (e) => {
      const t = e.touches[0];
      const rect = el.getBoundingClientRect();
      setMouse({ x: t.clientX - rect.left, y: t.clientY - rect.top });
    }, { passive: true });
    el.addEventListener('touchend', handleLeave);

    return () => {
      el.removeEventListener('mousemove', handleMove);
      el.removeEventListener('mouseleave', handleLeave);
      el.removeEventListener('touchmove', handleMove);
      el.removeEventListener('touchend', handleLeave);
    };
  }, []);

  // falloff curve
  const curve = (d) => {
    if (d == null) return 0;
    const t = Math.max(0, Math.min(1, 1 - d / radius)); // 0..1
    switch (falloff) {
      case 'exponential': return t * t;
      case 'gaussian': {
        // smooth bump: e^(-(d^2)/(2*sigma^2)), with sigma ~ radius/3
        const sigma = radius / 3;
        return Math.exp(-(d * d) / (2 * sigma * sigma));
      }
      default: return t; // linear
    }
  };

  // animate each letter via rAF for smoothness
  useEffect(() => {
    let raf;
    const tick = () => {
      spanRefs.current.forEach((span) => {
        if (!span) return;
        const rect = span.getBoundingClientRect();
        const parentRect = containerRef.current.getBoundingClientRect();
        const cx = rect.left - parentRect.left + rect.width / 2;
        const cy = rect.top - parentRect.top + rect.height / 2;

        const d = (mouse.x == null || mouse.y == null)
          ? Infinity
          : Math.hypot(mouse.x - cx, mouse.y - cy);

        const k = curve(d); // 0..1
        const w = Math.round(weightFrom + (weightTo - weightFrom) * k);
        const opsz = Math.round(opszFrom + (opszTo - opszFrom) * k);
        const alpha = minOpacity + (maxOpacity - minOpacity) * k;

        span.style.setProperty('fontVariationSettings', `'wght' ${w}, 'opsz' ${opsz}`);
        span.style.setProperty('opacity', alpha);
        span.style.setProperty('transform', `translateY(${(-4 * k).toFixed(2)}px)`); // tiny lift
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mouse.x, mouse.y, radius, falloff, weightFrom, weightTo, opszFrom, opszTo, minOpacity, maxOpacity]);

  return (
    <span
      ref={containerRef}
      className={`inline-block select-none leading-tight font-[\'Roboto_Flex\',Roboto,system-ui,sans-serif] ${className}`}
      style={{
        // base variation to avoid layout jumps when idle
        fontVariationSettings: `'wght' ${weightFrom}, 'opsz' ${opszFrom}`,
        cursor: 'default',
      }}
      aria-label={text}
    >
      {letters.map((ch, i) => (
        <span
          // store ref
          ref={(el) => (spanRefs.current[i] = el)}
          key={`${ch}-${i}`}
          style={{
            display: 'inline-block',
            transition: 'opacity 120ms ease, transform 120ms ease',
            willChange: 'font-variation-settings, opacity, transform',
          }}
        >
          {ch === ' ' ? '\u00A0' : ch}
        </span>
      ))}
    </span>
  );
}
