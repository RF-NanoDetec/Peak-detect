# Seamless Plot Design - Complete Redesign

## Overview
Complete redesign of the preprocessing visualization to blend seamlessly into the UI with no visual boundaries, making the plot feel like a natural part of the page rather than a separate component.

## Design Philosophy

### Core Principle: Transparency
The plot has **zero visual boundaries**. No cards, no borders, no shadows, no containers. The plot floats directly on the page background as if it's painted onto the canvas itself.

### Key Design Decisions

1. **Fully Transparent Backgrounds**
   - Plot background: `rgba(0,0,0,0)` (completely transparent)
   - Paper background: `rgba(0,0,0,0)` (completely transparent)
   - No container backgrounds or cards

2. **Subtle Grid Lines**
   - Dark mode: `rgba(255, 255, 255, 0.05)` - barely visible white grid
   - Light mode: `rgba(0, 0, 0, 0.06)` - barely visible black grid
   - Grid lines are hints, not boundaries

3. **Data Line Styling**
   - **Raw data**: Subtle and understated with reduced opacity
     - Dark mode: `rgba(148, 163, 184, 0.4)`
     - Light mode: `rgba(100, 116, 139, 0.4)`
     - Width: 1px
   - **Filtered data**: Vibrant and uses the theme primary color
     - Dark mode: `hsl(217.2, 91.2%, 59.8%)` - bright blue
     - Light mode: `hsl(221.2, 83.2%, 53.3%)` - rich blue
     - Width: 1.5px for emphasis

4. **Horizontal Legend**
   - Positioned above the plot, centered
   - Transparent background, no borders
   - Compact labels: "Original (3,103,575 pts)" and "Filtered (5,001 pts)"
   - Floats naturally with the plot

5. **Minimal Dataset Info**
   - Tiny text (11px) at top showing point counts
   - No card, no styling - just information
   - Example: "3,103,575 points • showing 5,001"

6. **No Modebar**
   - Completely hidden via CSS and config
   - No download button, no zoom controls
   - Pure visualization without chrome

## Technical Implementation

### Backend (`service/plotly_plotting.py`)

```python
# Transparent colors for seamless blending
if theme == "dark":
    bg_color = "rgba(0,0,0,0)"  # Fully transparent
    paper_color = "rgba(0,0,0,0)"  # Fully transparent
    text_color = "hsl(210, 40%, 98%)"  # --foreground
    grid_color = "rgba(255, 255, 255, 0.05)"  # Very subtle
    raw_color = "rgba(148, 163, 184, 0.4)"  # Subtle gray
    filtered_color = "hsl(217.2, 91.2%, 59.8%)"  # --primary
else:
    bg_color = "rgba(0,0,0,0)"  # Fully transparent
    paper_color = "rgba(0,0,0,0)"  # Fully transparent
    text_color = "hsl(222.2, 84%, 4.9%)"  # --foreground
    grid_color = "rgba(0, 0, 0, 0.06)"  # Very subtle
    raw_color = "rgba(100, 116, 139, 0.4)"  # Subtle gray
    filtered_color = "hsl(221.2, 83.2%, 53.3%)"  # --primary
```

**Layout Changes:**
- Removed title (redundant with page heading)
- Horizontal legend centered above plot
- Transparent legend background
- Smaller fonts (11px legend, 13px axis titles)
- No axis lines, only grid
- Tight margins: `{"l": 60, "r": 20, "t": 40, "b": 50}`

### Frontend (`ui-web/components/charts/PreprocessingPlotly.tsx`)

```tsx
return (
  <div className="flex flex-col h-full">
    {/* Minimal dataset info - no card */}
    <div className="px-8 pt-6 pb-2">
      <p className="text-[11px] text-muted-foreground">
        {original} points • showing {displayed}
      </p>
    </div>
    
    {/* Plot with no container */}
    <div className="flex-1 px-4">
      <Plot
        data={plotData.data}
        layout={{ ...plotData.layout, autosize: true, dragmode: false }}
        config={{ displayModeBar: false }}
        style={{ width: "100%", height: "100%" }}
        className="plotly-chart-seamless"
      />
    </div>
  </div>
)
```

**Key Points:**
- `flex-1` and `h-full` to fill parent container
- No card styling, no borders, no shadows
- `displayModeBar: false` to hide toolbar
- Custom CSS class for transparency enforcement

### CSS (`ui-web/app/globals.css`)

```css
/* Plotly seamless integration */
.plotly-chart-seamless {
  background: transparent !important;
}

.plotly-chart-seamless .main-svg {
  background: transparent !important;
}

.plotly-chart-seamless .modebar {
  display: none !important;
}
```

### Page Integration (`ui-web/app/preprocess/page.tsx`)

```tsx
<PageVisualization>
  {resultId ? (
    <PreprocessingPlotly
      resultId={resultId}
      filteredResultId={filteredResultId}
      className="flex-1"
    />
  ) : (
    // ... empty state
  )}
</PageVisualization>
```

No wrapper div - plot directly fills the PageVisualization area.

## Visual Results

### Light Mode
- Plot floats on white background
- Subtle black grid lines (6% opacity)
- Muted gray raw data line (40% opacity)
- Vibrant blue filtered data line
- Text in dark color matching theme
- Clean, airy, professional

### Dark Mode
- Plot floats on dark blue background (`hsl(222.2, 84%, 4.9%)`)
- Barely visible white grid lines (5% opacity)
- Subtle gray raw data line (40% opacity)
- Bright blue filtered data line
- Text in light color matching theme
- Elegant, modern, immersive

## Comparison: Before vs After

### Before (Old Design)
- ❌ White card container in dark mode (jarring contrast)
- ❌ Visible borders and shadows
- ❌ Separate "boxed" appearance
- ❌ Toolbar with many buttons
- ❌ Vertical legend taking space
- ❌ Title repeating page heading
- ❌ Dataset info in styled card

### After (New Design)
- ✅ Fully transparent, blends with page
- ✅ No borders, no shadows, no containers
- ✅ Feels like part of the page
- ✅ No toolbar at all
- ✅ Horizontal legend above plot
- ✅ No redundant title
- ✅ Minimal dataset info as plain text

## User Experience Benefits

1. **Visual Cohesion**: Plot feels like native UI, not embedded component
2. **Focus on Data**: No visual distractions from unnecessary chrome
3. **Theme Consistency**: Perfect adaptation to light/dark modes
4. **Professional Appearance**: Clean, modern, publication-quality
5. **Maximum Data**: No space wasted on decorative elements
6. **Accessibility**: Proper contrast and readable fonts maintained

## Smart Decimation Preserved

The intelligent decimation algorithm remains unchanged:
- Detects peaks using scipy
- Keeps peaks + 50-point context windows
- Uniformly samples background regions
- Reduces 300k+ points to ~5k while preserving features

## Interactive Features

**Kept:**
- ✅ Hover tooltips with exact values
- ✅ Legend toggle (click to show/hide traces)
- ✅ Responsive resizing
- ✅ GPU-accelerated rendering (scattergl)

**Removed:**
- ❌ Zoom and pan (inappropriate for decimated data)
- ❌ Download button (not needed in workflow)
- ❌ Modebar (visual clutter)
- ❌ Selection tools (not useful here)

## Performance

- Fast rendering with WebGL (scattergl traces)
- ~5,000 points displayed (from 300k+ original)
- Instant theme switching
- Smooth hover interactions
- Responsive to window resizing

## Code Quality

- Zero deprecated code or workarounds
- Clean separation of concerns
- Type-safe TypeScript
- Proper React hooks usage
- No inline styles (except required Plotly config)
- CSS classes follow Tailwind conventions

## Future Considerations

Possible enhancements:
1. Export functionality in a separate menu
2. Keyboard navigation for accessibility
3. Animation on data load
4. Touch gestures for mobile
5. Annotation tools if needed

## Conclusion

This redesign achieves the goal of making the plot **invisible as a component** while making the **data highly visible**. The plot is no longer something embedded in the page - it IS the page. This is modern, clean, professional data visualization that respects the user's attention and the application's design language.


