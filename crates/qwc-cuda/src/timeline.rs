//! Пофазная разбивка шага на событиях CUDA.
//!
//! Метка — это `cudaEventRecord` в поток; интервал между соседними метками
//! относится к фазе, которую открыла первая из них. Всё лежит на одном
//! потоке, поэтому сумма интервалов равна времени шага целиком: ни одна
//! миллисекунда не теряется, а пробелы от голодания по запускам попадают в ту
//! фазу, где они возникли.
//!
//! Стоимость самой записи — около микросекунды на метку на хосте, порядка
//! половины миллисекунды на шаг из полутысячи меток. Поэтому замер с
//! таймлайном сверяется с замером без него: разница и есть искажение.

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

    /// Начать шаг заново. События переиспользуются: их создание стоит дороже
    /// записи, а количество меток от шага к шагу одно и то же.
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

    /// Суммы по фазам в порядке первого появления. Последняя метка закрывает
    /// шаг и собственного интервала не имеет.
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

    /// Время от первой метки до последней.
    pub fn total_ms(&self) -> Result<f32> {
        if self.labels.len() < 2 {
            return Ok(0.0);
        }
        let last = self.labels.len() - 1;
        self.events[last].synchronize()?;
        Event::elapsed_ms(&self.events[0], &self.events[last])
    }
}
