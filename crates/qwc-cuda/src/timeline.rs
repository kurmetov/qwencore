//! Per-phase breakdown of a step using CUDA events.
//!
//! A mark is a `cudaEventRecord` into the stream; the interval between adjacent marks
//! belongs to the phase the first of them opened. Everything is on one
//! stream, so the sum of intervals equals the whole step time: not a single
//! millisecond is lost, and gaps from launch starvation land in the
//! phase where they occurred.
//!
//! The recording itself costs about a microsecond per mark on the host, on the order
//! of half a millisecond per step of five hundred marks. So a measurement with
//! the timeline is checked against one without it: the difference is the distortion.

use crate::error::Result;
use crate::stream::{Event, Stream};

#[derive(Default)]
pub struct Timeline {
    events: Vec<Event>,
    labels: Vec<&'static str>,
}

impl Timeline {
    pub fn new() -> Self {
        Self::default()
    }

    /// Restart the step. Events are reused: creating them costs more than
    /// recording, and the number of marks is the same from step to step.
    pub fn reset(&mut self) {
        self.labels.clear();
    }

    pub fn mark(&mut self, label: &'static str, stream: &Stream) -> Result<()> {
        let index = self.labels.len();
        if index == self.events.len() {
            self.events.push(Event::new()?);
        }
        self.events[index].record(stream)?;
        self.labels.push(label);
        Ok(())
    }

    pub fn marks(&self) -> usize {
        self.labels.len()
    }

    /// Per-phase sums in order of first appearance. The last mark closes
    /// the step and has no interval of its own.
    pub fn totals(&self) -> Result<Vec<(&'static str, f32)>> {
        if self.labels.len() < 2 {
            return Ok(Vec::new());
        }
        let last = self.labels.len() - 1;
        self.events[last].synchronize()?;

        let mut totals: Vec<(&'static str, f32)> = Vec::new();
        for index in 0..last {
            let ms = Event::elapsed_ms(&self.events[index], &self.events[index + 1])?;
            let label = self.labels[index];
            match totals.iter_mut().find(|(name, _)| *name == label) {
                Some((_, sum)) => *sum += ms,
                None => totals.push((label, ms)),
            }
        }
        Ok(totals)
    }

    /// Time from the first mark to the last.
    pub fn total_ms(&self) -> Result<f32> {
        if self.labels.len() < 2 {
            return Ok(0.0);
        }
        let last = self.labels.len() - 1;
        self.events[last].synchronize()?;
        Event::elapsed_ms(&self.events[0], &self.events[last])
    }
}
