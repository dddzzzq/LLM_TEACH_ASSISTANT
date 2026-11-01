root => {
                    const bg = root.querySelector('canvas');
                    const block = root.querySelector('canvas.block');
                    if (!bg || !block || bg === block) throw Error('Unsupported slider');
                    return {bg: bg.toDataURL('image/png'), block: block.toDataURL('image/png'),
                            scale: bg.getBoundingClientRect().width / bg.width};
                }