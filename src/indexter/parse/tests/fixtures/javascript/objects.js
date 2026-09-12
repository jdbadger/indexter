const first = {
  handler(event) {
    return event.type;
  },
};

const second = {
  handler(event) {
    return event.target;
  },
};
