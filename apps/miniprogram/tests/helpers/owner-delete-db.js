// Synthetic serialized CloudBase transactions with isolated reads and rollback.
const clone = value => JSON.parse(JSON.stringify(value))
function ownerDeleteDb(seed = {}, { failWrite, strictTransactions = false, beforeCommit, retryConflicts = 0, queryIds = false } = {}) {
  const rows = clone(seed)
  let queue = Promise.resolve()
  const transactionAttempts = []
  function collection(name, attempt = null) {
    function call(method, id) {
      if (!attempt) return
      attempt.calls.push({ collection: name, method, id }); attempt.count++
      if (strictTransactions && attempt.count > 100) { attempt.error = new Error('transaction operation limit exceeded'); throw attempt.error }
    }
    function query(method) {
      if (attempt && strictTransactions) { attempt.error = new Error(`transaction query unsupported: ${method}`); throw attempt.error }
    }
    rows[name] ||= {}
    let filter = {}; const order = []; let limit = Infinity
    function matches(row, value) {
      if (value.op === 'and') return value.values.every(item => matches(row, item))
      if (value.op === 'or') return value.values.some(item => matches(row, item))
      return Object.entries(value).every(([key, item]) => {
        const field = key.split('.').reduce((current, part) => current?.[part], row)
        if (item && typeof item === 'object') {
          if (item.op === 'elemMatch') return Array.isArray(field) && field.some(entry => matches(entry, item.value))
          return item.op === 'gt' ? field > item.value : item.op === 'lt' ? field < item.value : field <= item.lte
        }
        return field === item
      })
    }
    return {
      doc(id) { return {
        async get() { call('get', id); return { data: clone(rows[name][id] ?? null) } },
        async set({ data }) { call('set', id); if (failWrite) failWrite(name, id); rows[name][id] = clone(data) },
        async update({ data }) { call('update', id); if (failWrite) failWrite(name, id); Object.assign(rows[name][id], clone(data)) },
        async remove() { call('remove', id); if (failWrite) failWrite(name, id); delete rows[name][id] },
      } },
      where(value) { query('where'); filter = value; return this }, orderBy(key, direction) { query('orderBy'); order.push([key, direction]); return this },
      limit(value) { query('limit'); limit = value; return this },
      aggregate() { query('aggregate'); throw new Error('aggregate unavailable') },
      count() { query('count'); return Promise.resolve({ total: Object.values(rows[name]).filter(row => matches(row, filter)).length }) },
      async get() { query('get'); return { data: Object.entries(rows[name]).map(([key,row]) => queryIds ? { ...row, _id:key } : row).filter(row => matches(row, filter)).sort((a, b) => {
        for (const [key, direction] of order) {
          const d = a[key] < b[key] ? -1 : a[key] > b[key] ? 1 : 0
          if (d) return direction === 'desc' ? -d : d
        }
        return 0
      }).slice(0, limit).map(row => clone(row)) } },
    }
  }
  return { rows, collection, transactionAttempts, setStrictTransactions(value) { strictTransactions = value }, command: {
    gt: value => ({ op: 'gt', value }), lte: value => ({ lte: value }), lt: value => ({ op: 'lt', value }),
    and: values => ({ op: 'and', values }), or: values => ({ op: 'or', values }),
    elemMatch: value => ({ op: 'elemMatch', value }),
  }, async runTransaction(callback) {
    const task = queue.then(async () => {
      for (let index = 0; ; index++) {
        const snapshot = clone(rows); const attempt = { count: 0, calls: [], index }; transactionAttempts.push(attempt)
        try {
          const result = await callback({ collection: name => collection(name, attempt) })
          if (attempt.error) throw attempt.error
          if (beforeCommit) await beforeCommit({ attempt, rows })
          return result
        } catch (error) {
          for (const name of Object.keys(rows)) delete rows[name]
          Object.assign(rows, snapshot)
          if (error.code === 'SYNTHETIC_CONFLICT' && index < retryConflicts) continue
          throw error
        }
      }
    })
    queue = task.catch(() => {})
    return task
  } }
}
module.exports = { ownerDeleteDb }
