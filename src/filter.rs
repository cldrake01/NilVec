use std::sync::Arc;

use crate::metadata::Metadata;

/// A filter that checks an attribute’s metadata.
pub struct Filter {
    pub attribute: String,
    // Use Arc so that the closure is cloneable, and require 'static.
    pub condition: Arc<dyn Fn(&Metadata) -> bool + Send + Sync + 'static>,
}

// Manually implement Clone (though Arc is cloneable)
impl Clone for Filter {
    fn clone(&self) -> Self {
        Filter {
            attribute: self.attribute.clone(),
            condition: self.condition.clone(),
        }
    }
}
