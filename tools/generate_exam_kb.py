import json
from itertools import cycle

# Configuration: how many topics per subject
counts = {
    'physics': 180,
    'math': 200,
    'chemistry': 120
}

# Base topic templates per subject
physics_bases = [
    'Kinematics: {}', 'Dynamics: {}', 'Work and Energy: {}', 'Momentum: {}', 'Circular Motion: {}',
    'Rotational Dynamics: {}', 'Gravitation: {}', 'Oscillation: {}', 'Mechanical Waves: {}', 'Sound: {}',
    'Thermodynamics: {}', 'Heat Transfer: {}', 'Electrostatics: {}', 'Electric Circuits: {}', 'Magnetism: {}',
    'Electromagnetic Induction: {}', 'Optics: {}', 'Geometric Optics: {}', 'Wave Optics: {}', 'Modern Physics: {}'
]

math_bases = [
    'Algebra: {}', 'Quadratics: {}', 'Polynomials: {}', 'Functions: {}', 'Exponents and Logarithms: {}',
    'Trigonometry: {}', 'Trigonometric Identities: {}', 'Analytic Geometry: {}', 'Vectors: {}', 'Matrices: {}',
    'Sequences and Series: {}', 'Limits: {}', 'Derivatives: {}', 'Integrals: {}', 'Differential Equations: {}',
    'Probability: {}', 'Statistics: {}', 'Coordinate Geometry: {}', 'Complex Numbers: {}', 'Number Theory: {}'
]

chem_bases = [
    'Stoichiometry: {}', 'Atomic Structure: {}', 'Periodic Table: {}', 'Chemical Bonding: {}', 'Molecular Geometry: {}',
    'Thermochemistry: {}', 'Chemical Kinetics: {}', 'Chemical Equilibrium: {}', 'Acids and Bases: {}', 'Redox Reactions: {}',
    'Organic Basics: {}', 'Hydrocarbons: {}', 'Functional Groups: {}', 'Stoichiometric Calculations: {}', 'Gas Laws: {}'
]

# Helper to produce plausible keywords and examples
def keywords_for_subject(subject, idx):
    if subject == 'physics':
        return ['斜面','力','加速度','速度','位移','能量']
    if subject == 'math':
        return ['方程','函数','证明','解法','变换','性质']
    if subject == 'chemistry':
        return ['化学式','配平','摩尔','反应','酸碱','键']
    return []


def examples_for_subject(subject, idx):
    if subject == 'physics':
        return [f'例题：分析第{idx}题斜面受力与加速度计算']
    if subject == 'math':
        return [f'例题：求第{idx}题的解或证明步骤']
    if subject == 'chemistry':
        return [f'例题：计算第{idx}题的产物质量或平衡常数']
    return []


def make_topics(subject, count, bases):
    topics = []
    base_cycle = cycle(bases)
    i = 1
    for _ in range(count):
        base = next(base_cycle)
        # create a descriptive suffix
        suffix = f'基本概念 {i}'
        title = base.format(suffix)
        tid = f'{subject}_{i:03d}'
        aliases = [title.split(':')[0], title.replace(' ', '')]
        level = ['高中', subject.capitalize()]
        description = f'本知识点为{subject}专题：{title}，包含基础概念、常见题型与解题方法。'
        keywords = keywords_for_subject(subject, i)
        examples = examples_for_subject(subject, i)
        topics.append({
            'id': tid,
            'title': title,
            'aliases': aliases,
            'level': level,
            'description': description,
            'keywords': keywords,
            'examples': examples,
            'embedding': None
        })
        i += 1
    return topics


def main():
    kb = {'subjects': {}}
    kb['subjects']['physics'] = {
        'display_name': 'Physics',
        'topics': make_topics('physics', counts['physics'], physics_bases)
    }
    kb['subjects']['math'] = {
        'display_name': 'Mathematics',
        'topics': make_topics('math', counts['math'], math_bases)
    }
    kb['subjects']['chemistry'] = {
        'display_name': 'Chemistry',
        'topics': make_topics('chemistry', counts['chemistry'], chem_bases)
    }

    # Write file next to the project root (script is in tools/)
    import os
    out_path = os.path.join(os.path.dirname(__file__), '..', 'exam_kb.json')
    out_path = os.path.normpath(out_path)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    print('Generated exam_kb.json with counts:', {k: len(v['topics']) for k,v in kb['subjects'].items()})

if __name__ == '__main__':
    main()
